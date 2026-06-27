// Переиспользуемый Vulkan compute backend — фундамент собственного SD-движка.
// VkCtx: один раз инициализирует device/queue/pools. Kernel: pipeline из SPIR-V.
// На этом слое строятся все ядра (matmul, conv2d, attention, norm...).
#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "VkEngine", __VA_ARGS__)

struct Buf { VkBuffer buf=VK_NULL_HANDLE; VkDeviceMemory mem=VK_NULL_HANDLE; VkDeviceSize size=0; };

// ---------- Контекст ----------
struct VkCtx {
    VkInstance inst=VK_NULL_HANDLE;
    VkPhysicalDevice phys=VK_NULL_HANDLE;
    VkDevice dev=VK_NULL_HANDLE;
    VkQueue queue=VK_NULL_HANDLE;
    uint32_t qfi=0;
    VkCommandPool pool=VK_NULL_HANDLE;
    bool fp16=false;

    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
        if (vkCreateInstance(&ici,nullptr,&inst)!=VK_SUCCESS) return false;
        uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr);
        std::vector<VkPhysicalDevice> ds(nd); vkEnumeratePhysicalDevices(inst,&nd,ds.data());
        phys=ds[0];
        // subgroup (warp) свойства
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 pr2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; pr2.pNext=&sg;
        vkGetPhysicalDeviceProperties2(phys,&pr2);
        VkPhysicalDeviceProperties& pr=pr2.properties; LOG("GPU: %s", pr.deviceName);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
        VkDeviceSize vram=0; for(uint32_t i=0;i<mp.memoryHeapCount;i++) if(mp.memoryHeaps[i].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram+=mp.memoryHeaps[i].size;
        LOG("HWCAPS api=%u.%u.%u vendor=0x%x device=0x%x | sharedMem=%uKB wgInvoc=%u wgSize=%ux%ux%u | subgroup=%u | maxAlloc=%uMB vram=%lluMB",
            VK_VERSION_MAJOR(pr.apiVersion),VK_VERSION_MINOR(pr.apiVersion),VK_VERSION_PATCH(pr.apiVersion),
            pr.vendorID, pr.deviceID,
            pr.limits.maxComputeSharedMemorySize/1024u, pr.limits.maxComputeWorkGroupInvocations,
            pr.limits.maxComputeWorkGroupSize[0],pr.limits.maxComputeWorkGroupSize[1],pr.limits.maxComputeWorkGroupSize[2],
            sg.subgroupSize, (uint32_t)(pr.limits.maxStorageBufferRange/1048576u), (unsigned long long)(vram/1048576ull));
        uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&nq,nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys,&nq,qf.data());
        for (uint32_t i=0;i<nq;i++) if (qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ qfi=i; break; }
        // fp16 features
        VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDevice16BitStorageFeatures s16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES}; f16.pNext=&s16;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext=&f16;
        vkGetPhysicalDeviceFeatures2(phys,&f2); fp16=f16.shaderFloat16 && s16.storageBuffer16BitAccess;
        float prio=1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex=qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.pNext=&f2;
        if (vkCreateDevice(phys,&dci,nullptr,&dev)!=VK_SUCCESS) return false;
        vkGetDeviceQueue(dev,qfi,0,&queue);
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cp.queueFamilyIndex=qfi;
        vkCreateCommandPool(dev,&cp,nullptr,&pool);
        return true;
    }
    void destroy(){ if(dev){vkDeviceWaitIdle(dev);vkDestroyCommandPool(dev,pool,nullptr);vkDestroyDevice(dev,nullptr);} if(inst)vkDestroyInstance(inst,nullptr); }

    uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want){
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
        for (uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want) return i;
        return UINT32_MAX;
    }
    Buf alloc(VkDeviceSize sz, bool deviceLocal){
        Buf b; b.size=sz;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size=sz; bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(dev,&bi,nullptr,&b.buf);
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev,b.buf,&rq);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=rq.size;
        VkMemoryPropertyFlags want=deviceLocal?VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            :(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ai.memoryTypeIndex=memType(rq.memoryTypeBits,want);
        vkAllocateMemory(dev,&ai,nullptr,&b.mem); vkBindBufferMemory(dev,b.buf,b.mem,0);
        return b;
    }
    void free(Buf& b){ if(b.buf)vkDestroyBuffer(dev,b.buf,nullptr); if(b.mem)vkFreeMemory(dev,b.mem,nullptr); b={}; }

    VkCommandBuffer beginCmd(){
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=pool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VkCommandBuffer c; vkAllocateCommandBuffers(dev,&ai,&c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
        return c;
    }
    void endCmd(VkCommandBuffer c){
        vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c;
        vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(dev,pool,1,&c);
    }
    // staging upload host→device-local
    void upload(Buf& dst, const void* src, VkDeviceSize sz){
        Buf st=alloc(sz,false); void* p; vkMapMemory(dev,st.mem,0,sz,0,&p); memcpy(p,src,sz); vkUnmapMemory(dev,st.mem);
        VkCommandBuffer c=beginCmd(); VkBufferCopy cp{0,0,sz}; vkCmdCopyBuffer(c,st.buf,dst.buf,1,&cp); endCmd(c); free(st);
    }
    void download(Buf& src, void* dst, VkDeviceSize sz){
        Buf st=alloc(sz,false);
        VkCommandBuffer c=beginCmd(); VkBufferCopy cp{0,0,sz}; vkCmdCopyBuffer(c,src.buf,st.buf,1,&cp); endCmd(c);
        void* p; vkMapMemory(dev,st.mem,0,sz,0,&p); memcpy(dst,p,sz); vkUnmapMemory(dev,st.mem); free(st);
    }
};

// ---------- Ядро (pipeline из SPIR-V) ----------
struct Kernel {
    VkDescriptorSetLayout dsl=VK_NULL_HANDLE;
    VkPipelineLayout pl=VK_NULL_HANDLE;
    VkPipeline pipe=VK_NULL_HANDLE;
    VkShaderModule sm=VK_NULL_HANDLE;
    VkDescriptorPool dp=VK_NULL_HANDLE;
    int nBuf=0; uint32_t pcBytes=0;

    void create(VkCtx& c, const void* spv, size_t spvLen, int numBuffers, uint32_t pushBytes, int poolSets=1){
        nBuf=numBuffers; pcBytes=pushBytes;
        VkShaderModuleCreateInfo sm_{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm_.codeSize=spvLen; sm_.pCode=(const uint32_t*)spv; vkCreateShaderModule(c.dev,&sm_,nullptr,&sm);
        std::vector<VkDescriptorSetLayoutBinding> b(numBuffers);
        for (int i=0;i<numBuffers;i++){ b[i]={(uint32_t)i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}; }
        VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        di.bindingCount=numBuffers; di.pBindings=b.data(); vkCreateDescriptorSetLayout(c.dev,&di,nullptr,&dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,pushBytes};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount=1; pli.pSetLayouts=&dsl;
        if (pushBytes){ pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; }
        vkCreatePipelineLayout(c.dev,&pli,nullptr,&pl);
        VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage=ss; cp.layout=pl;
        vkCreateComputePipelines(c.dev,VK_NULL_HANDLE,1,&cp,nullptr,&pipe);
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(uint32_t)numBuffers*poolSets};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets=poolSets; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; vkCreateDescriptorPool(c.dev,&dpi,nullptr,&dp);
    }
    void resetPool(VkCtx& c){ vkResetDescriptorPool(c.dev,dp,0); }
    VkDescriptorSet makeSet(VkCtx& c, std::vector<Buf*> bufs){
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=dp; ai.descriptorSetCount=1; ai.pSetLayouts=&dsl;
        VkDescriptorSet ds; vkAllocateDescriptorSets(c.dev,&ai,&ds);
        std::vector<VkDescriptorBufferInfo> bi(bufs.size());
        std::vector<VkWriteDescriptorSet> wr(bufs.size());
        for (size_t i=0;i<bufs.size();i++){
            bi[i]={bufs[i]->buf,0,bufs[i]->size};
            wr[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet=ds; wr[i].dstBinding=(uint32_t)i;
            wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo=&bi[i];
        }
        vkUpdateDescriptorSets(c.dev,(uint32_t)wr.size(),wr.data(),0,nullptr);
        return ds;
    }
    void record(VkCommandBuffer cmd, VkDescriptorSet ds, const void* push, uint32_t gx, uint32_t gy, uint32_t gz){
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
        if (pcBytes) vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,pcBytes,push);
        vkCmdDispatch(cmd,gx,gy,gz);
    }
    void destroy(VkCtx& c){
        vkDestroyDescriptorPool(c.dev,dp,nullptr); vkDestroyPipeline(c.dev,pipe,nullptr);
        vkDestroyPipelineLayout(c.dev,pl,nullptr); vkDestroyDescriptorSetLayout(c.dev,dsl,nullptr);
        vkDestroyShaderModule(c.dev,sm,nullptr);
    }
};

static double timeDispatch(VkCtx& c, Kernel& k, VkDescriptorSet ds, const void* push, uint32_t gx,uint32_t gy,uint32_t gz, int iters){
    auto one=[&](){ VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,push,gx,gy,gz);
        vkEndCommandBuffer(cmd); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        vkQueueSubmit(c.queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(c.queue); vkFreeCommandBuffers(c.dev,c.pool,1,&cmd); };
    one(); // прогрев
    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<iters;i++) one();
    auto t1=std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1-t0).count()/iters;
}

// ====================== JNI: MATMUL ======================
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchMatmul(
        JNIEnv* env, jobject, jbyteArray spirv, jint M, jint N, jint K, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    jsize spvLen=env->GetArrayLength(spirv); std::vector<uint8_t> spv(spvLen);
    env->GetByteArrayRegion(spirv,0,spvLen,(jbyte*)spv.data());

    VkDeviceSize szA=(VkDeviceSize)M*K*4, szB=(VkDeviceSize)K*N*4, szC=(VkDeviceSize)M*N*4;  // fp32
    Buf bA=c.alloc(szA,true), bB=c.alloc(szB,true), bC=c.alloc(szC,true);
    std::vector<float> hA((size_t)M*K), hB((size_t)K*N);
    for (size_t i=0;i<hA.size();i++) hA[i]=(((i*131u+7u)%17u)*0.01f-0.08f);
    for (size_t i=0;i<hB.size();i++) hB[i]=(((i*61u+13u)%19u)*0.01f-0.09f);
    c.upload(bA,hA.data(),szA); c.upload(bB,hB.data(),szB);

    Kernel k; k.create(c,spv.data(),spvLen,3,20);
    VkDescriptorSet ds=k.makeSet(c,{&bA,&bB,&bC});
    uint32_t pc[5]={(uint32_t)M,(uint32_t)N,(uint32_t)K,(uint32_t)N,0u};
    uint32_t gx=((uint32_t)N+127)/128, gy=((uint32_t)M+127)/128;

    // прогрев + верификация (fp32)
    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,pc,gx,gy,1); c.endCmd(cmd); }
    { std::vector<float> hC((size_t)M*N); c.download(bC,hC.data(),szC);
      double se=0,sr=0; for(int s=0;s<16;s++){ int r=(s*97+3)%M,col=(s*53+11)%N; float ref=0;
        for(int kk=0;kk<K;kk++) ref+=hA[(size_t)r*K+kk]*hB[(size_t)kk*N+col];
        se+=fabsf(ref-hC[(size_t)r*N+col]); sr+=fabsf(ref);} double e=se/(sr+1e-6);
      LOG("MATMUL %dx%dx%d corr=%.4f %s",M,N,K,e,e<0.02?"OK":"FAIL"); }

    double sec=timeDispatch(c,k,ds,pc,gx,gy,1,iters);
    double g=2.0*(double)M*N*K/sec/1e9;
    LOG("matmul %dx%dx%d: %.3f ms, %.1f GFLOPS",M,N,K,sec*1000,g);
    k.destroy(c); c.free(bA);c.free(bB);c.free(bC); c.destroy();
    return g;
}

// ====================== JNI: SILU ======================
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchSilu(
        JNIEnv* env, jobject, jbyteArray spirv, jint n, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    jsize l=env->GetArrayLength(spirv); std::vector<uint8_t> spv(l); env->GetByteArrayRegion(spirv,0,l,(jbyte*)spv.data());
    VkDeviceSize sz=(VkDeviceSize)n*2; Buf bX=c.alloc(sz,true), bY=c.alloc(sz,true);
    std::vector<__fp16> hX(n); for(int i=0;i<n;i++) hX[i]=(__fp16)(((i*131u+7u)%37u)*0.1f-1.8f);
    c.upload(bX,hX.data(),sz);
    Kernel k; k.create(c,spv.data(),spv.size(),2,4); VkDescriptorSet ds=k.makeSet(c,{&bX,&bY});
    uint32_t pc=(uint32_t)n; uint32_t gx=((uint32_t)n+255)/256;
    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,&pc,gx,1,1); c.endCmd(cmd); }
    { std::vector<__fp16> hY(n); c.download(bY,hY.data(),sz); double se=0,sr=0;
      for(int s=0;s<32;s++){ int i=(s*131+5)%n; float x=(float)hX[i]; float ref=x/(1.0f+expf(-x));
        se+=fabsf(ref-(float)hY[i]); sr+=fabsf(ref);} double e=se/(sr+1e-6);
      LOG("SILU n%d corr=%.4f %s",n,e,e<0.02?"OK":"FAIL"); }
    double sec=timeDispatch(c,k,ds,&pc,gx,1,1,iters);
    LOG("silu n%d: %.3f ms",n,sec*1000);
    k.destroy(c); c.free(bX);c.free(bY); c.destroy(); return sec*1000;
}

// ====================== JNI: GROUPNORM ======================
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchGroupNorm(
        JNIEnv* env, jobject, jbyteArray spirv, jint C, jint HW, jint G, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    jsize l=env->GetArrayLength(spirv); std::vector<uint8_t> spv(l); env->GetByteArrayRegion(spirv,0,l,(jbyte*)spv.data());
    VkDeviceSize szX=(VkDeviceSize)C*HW*2, szP=(VkDeviceSize)C*2;
    Buf bX=c.alloc(szX,true), bG=c.alloc(szP,true), bB=c.alloc(szP,true), bY=c.alloc(szX,true);
    std::vector<__fp16> hX((size_t)C*HW), hG(C), hB(C);
    for(size_t i=0;i<hX.size();i++) hX[i]=(__fp16)(((i*131u+7u)%29u)*0.1f-1.4f);
    for(int i=0;i<C;i++){ hG[i]=(__fp16)(0.8f+0.01f*(i%7)); hB[i]=(__fp16)(0.05f*(i%5)-0.1f); }
    c.upload(bX,hX.data(),szX); c.upload(bG,hG.data(),szP); c.upload(bB,hB.data(),szP);
    Kernel k; k.create(c,spv.data(),spv.size(),4,16); VkDescriptorSet ds=k.makeSet(c,{&bX,&bG,&bB,&bY});
    struct { uint32_t C,HW,G; float eps; } pc{(uint32_t)C,(uint32_t)HW,(uint32_t)G,1e-5f};
    uint32_t gx=(uint32_t)G;
    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,&pc,gx,1,1); c.endCmd(cmd); }
    // верификация: пересчёт group-статистики на CPU
    { std::vector<__fp16> hY((size_t)C*HW); c.download(bY,hY.data(),szX);
      int cpg=C/G; int gs=cpg*HW; double se=0,sr=0;
      for (int gi=0; gi<G; gi++){
          double s=0,sq=0; int c0=gi*cpg;
          for(int j=0;j<gs;j++){ int ch=c0+j/HW, off=j%HW; float v=(float)hX[(size_t)ch*HW+off]; s+=v; sq+=v*v; }
          double mean=s/gs, var=sq/gs-mean*mean, inv=1.0/sqrt(var+1e-5);
          for (int t=0;t<4;t++){ int j=(t*97+gi)%gs; int ch=c0+j/HW, off=j%HW;
            float v=(float)hX[(size_t)ch*HW+off];
            float ref=(float)((v-mean)*inv*(float)hG[ch]+(float)hB[ch]);
            se+=fabsf(ref-(float)hY[(size_t)ch*HW+off]); sr+=fabsf(ref); } }
      double e=se/(sr+1e-6); LOG("GROUPNORM C%d HW%d G%d corr=%.4f %s",C,HW,G,e,e<0.03?"OK":"FAIL"); }
    double sec=timeDispatch(c,k,ds,&pc,gx,1,1,iters);
    LOG("groupnorm C%d HW%d: %.3f ms",C,HW,sec*1000);
    k.destroy(c); c.free(bX);c.free(bG);c.free(bB);c.free(bY); c.destroy(); return sec*1000;
}

// ====================== JNI: RESNET BLOCK (сборка из ядер, сверка с PyTorch) ======================
// shaders[6]: groupnorm, silu, conv2d, matmul, addbias, add
// weights[13]: x,temb,n1w,n1b,c1w,c1b,tprojWT,tprojB,n2w,n2b,c2w,c2b,y_ref
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_runResnetBlock(
        JNIEnv* env, jobject, jobjectArray shaders, jobjectArray weights, jint C, jint H, jint W, jint Temb) {
    VkCtx c; if(!c.init()) return -1;
    auto getArr=[&](jobjectArray a, int i){ jbyteArray e=(jbyteArray)env->GetObjectArrayElement(a,i);
        jsize l=env->GetArrayLength(e); std::vector<uint8_t> v(l); env->GetByteArrayRegion(e,0,l,(jbyte*)v.data()); return v; };
    std::vector<std::vector<uint8_t>> sh(6), w(13);
    for (int i=0;i<6;i++) sh[i]=getArr(shaders,i);
    for (int i=0;i<13;i++) w[i]=getArr(weights,i);

    uint32_t HW=(uint32_t)H*W, N=(uint32_t)C*HW, G=32;
    auto mkW=[&](int i, VkDeviceSize bytes){ Buf b=c.alloc(bytes,true); c.upload(b,w[i].data(),bytes); return b; };
    Buf bx=mkW(0,(VkDeviceSize)N*2), btemb=mkW(1,(VkDeviceSize)Temb*2);
    Buf n1w=mkW(2,C*2),n1b=mkW(3,C*2), c1w=mkW(4,(VkDeviceSize)C*C*9*2),c1b=mkW(5,C*2);
    Buf tpw=mkW(6,(VkDeviceSize)Temb*C*2),tpb=mkW(7,C*2);
    Buf n2w=mkW(8,C*2),n2b=mkW(9,C*2), c2w=mkW(10,(VkDeviceSize)C*C*9*2),c2b=mkW(11,C*2);
    std::vector<__fp16> yref(N); { auto& yv=w[12]; memcpy(yref.data(),yv.data(),(size_t)N*2); }

    // промежуточные
    Buf h=c.alloc((VkDeviceSize)N*2,true), hc=c.alloc((VkDeviceSize)N*2,true);
    Buf h2=c.alloc((VkDeviceSize)N*2,true), out=c.alloc((VkDeviceSize)N*2,true);
    Buf ts=c.alloc((VkDeviceSize)Temb*2,true), tp=c.alloc((VkDeviceSize)C*2,true);

    // универсальный запуск ядра (создать pipeline, dispatch, уничтожить)
    auto op=[&](int si, std::vector<Buf*> bufs, const void* push, uint32_t pb, uint32_t gx,uint32_t gy,uint32_t gz){
        Kernel k; k.create(c,sh[si].data(),sh[si].size(),(int)bufs.size(),pb);
        VkDescriptorSet ds=k.makeSet(c,bufs);
        VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,push,gx,gy,gz); c.endCmd(cmd); k.destroy(c);
    };
    struct GN{uint32_t C,HW,G;float eps;}; struct CV{uint32_t Ci,Co,H,W,KH,KW,pad,stride;};
    struct MM{uint32_t M,N,K;}; struct AB{uint32_t C,HW;}; struct N1{uint32_t n;};
    uint32_t gSilu=(N+255)/256, gAB=(N+255)/256;
    GN gn{(uint32_t)C,HW,G,1e-5f}; CV cv{(uint32_t)C,(uint32_t)C,(uint32_t)H,(uint32_t)W,3,3,1,1};
    uint32_t cgx=((uint32_t)W+15)/16, cgy=((uint32_t)H+15)/16, cgz=(uint32_t)C;

    // --- граф ResNet ---
    op(0,{&bx,&n1w,&n1b,&h},&gn,16,G,1,1);              // h = groupnorm(x)
    { N1 n{N}; op(1,{&h,&h},&n,4,gSilu,1,1); }          // h = silu(h)  (in-place)
    op(2,{&h,&c1w,&hc},&cv,32,cgx,cgy,cgz);             // hc = conv1(h)
    { AB ab{(uint32_t)C,HW}; op(4,{&hc,&c1b},&ab,8,gAB,1,1); }  // hc += conv1_bias
    // time embedding: ts=silu(temb); tp = ts @ tprojWT + tpb
    { N1 n{(uint32_t)Temb}; op(1,{&btemb,&ts},&n,4,((uint32_t)Temb+255)/256,1,1); }
    { MM mm{1,(uint32_t)C,(uint32_t)Temb}; op(3,{&ts,&tpw,&tp},&mm,12,((uint32_t)C+127)/128,1,1); }
    { AB ab{(uint32_t)C,1}; op(4,{&tp,&tpb},&ab,8,((uint32_t)C+255)/256,1,1); }  // tp += tproj_bias
    { AB ab{(uint32_t)C,HW}; op(4,{&hc,&tp},&ab,8,gAB,1,1); }   // hc += tp (broadcast по каналам)
    op(0,{&hc,&n2w,&n2b,&h2},&gn,16,G,1,1);             // h2 = groupnorm(hc)
    { N1 n{N}; op(1,{&h2,&h2},&n,4,gSilu,1,1); }        // h2 = silu(h2)
    op(2,{&h2,&c2w,&h},&cv,32,cgx,cgy,cgz);             // h = conv2(h2)  (переиспользуем h)
    { AB ab{(uint32_t)C,HW}; op(4,{&h,&c2b},&ab,8,gAB,1,1); }   // h += conv2_bias
    { N1 n{N}; op(5,{&bx,&h,&out},&n,4,gSilu,1,1); }    // out = x + h

    // сверка
    std::vector<__fp16> hy(N); c.download(out,hy.data(),(VkDeviceSize)N*2);
    double se=0,sr=0; for (uint32_t i=0;i<N;i+=37){ se+=fabsf((float)yref[i]-(float)hy[i]); sr+=fabsf((float)yref[i]); }
    double e=se/(sr+1e-6);
    LOG("RESNET C%d %dx%d corr=%.4f %s",C,H,W,e,e<0.03?"OK":"FAIL");
    c.destroy(); return e;
}

// ====================== ГРАФ-ДВИЖОК UNet ======================
#include <string>
#include <map>
#include <cstdio>
#include <cerrno>
namespace unet {
// шейдеры по индексу (порядок задаёт Kotlin)
enum S { GN=0, SILU, CONV, MM, AB, AB2, ADD, LN, SPLIT, ATTN, MERGE, GEGLU, T_CH, T_HC, UP, ATTN_BIG, IM2COL, NSH };
static VkCtx* C_; static std::vector<std::vector<uint8_t>>* SH_; static std::string DIR_;
static std::map<std::string,Buf> WC_;  // кэш весов (персистентный)
static std::vector<Buf> SCRATCH_;      // временные буферы forward (освобождаются после)

static int64_t fsize(const std::string& p){ FILE* f=fopen(p.c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); int64_t s=ftell(f); fclose(f); return s; }
// загрузка веса по имени (lazy, device-local)
static Buf& W(const std::string& name){
    auto it=WC_.find(name); if(it!=WC_.end()) return it->second;
    std::string p=DIR_+"/"+name+".bin"; int64_t sz=fsize(p);
    if (sz<0){ LOG("WEIGHT MISSING %s",name.c_str()); }
    // файлы fp16 → конвертируем в fp32 на GPU (точность critical-path)
    int64_t n=(sz>0?sz:2)/2; std::vector<__fp16> h(n);
    if (sz>0){ FILE* f=fopen(p.c_str(),"rb"); fread(h.data(),2,n,f); fclose(f); }
    std::vector<float> f32(n); for(int64_t i=0;i<n;i++) f32[i]=(float)h[i];
    Buf b=C_->alloc((VkDeviceSize)n*4,true); C_->upload(b,f32.data(),(VkDeviceSize)n*4);
    WC_[name]=b; return WC_[name];
}
static bool has(const std::string& name){ return fsize(DIR_+"/"+name+".bin")>0; }
static Buf mk(VkDeviceSize bytes){ Buf b=C_->alloc(bytes,true); SCRATCH_.push_back(b); return b; }

// --- single-command-buffer движок: pipeline'ы кэшируются (16 шейдеров создаём 1 раз),
//     все dispatch'и пишутся в один gCmd с барьерами памяти, submit/waitIdle 1 раз за forward ---
static Kernel gK[NSH]; static bool gKi[NSH]={false};
static VkCommandBuffer gCmd=VK_NULL_HANDLE;
static const int POOL_SETS=1024;  // макс. дескрипторов одного шейдера за forward (с запасом)
static Kernel& kern(int si,int nBuf,uint32_t pb){
    if(!gKi[si]){ gK[si].create(*C_,(*SH_)[si].data(),(*SH_)[si].size(),nBuf,pb,POOL_SETS); gKi[si]=true; }
    return gK[si];
}
// барьер write→read между операциями (compute+transfer, консервативно = идентично прежней сериализации)
static void barrier(){
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(gCmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb,0,nullptr,0,nullptr);
}
// профилировщик: каждая категория ops — суммарное GPU-время (standalone submit+waitIdle)
static bool gProfile=false; static double gProfT[NSH]={0}; static int gProfN[NSH]={0};
static void op(int si,std::vector<Buf*> bufs,const void* push,uint32_t pb,uint32_t gx,uint32_t gy,uint32_t gz){
    Kernel& k=kern(si,(int)bufs.size(),pb);
    VkDescriptorSet ds=k.makeSet(*C_,bufs);
    if (gProfile){
        VkCommandBuffer cmd=C_->beginCmd(); k.record(cmd,ds,push,gx,gy,gz);
        auto t0=std::chrono::high_resolution_clock::now(); C_->endCmd(cmd);
        auto t1=std::chrono::high_resolution_clock::now();
        gProfT[si]+=std::chrono::duration<double>(t1-t0).count(); gProfN[si]++; return;
    }
    k.record(gCmd,ds,push,gx,gy,gz); barrier();
}
struct GNp{uint32_t C,HW,G;float e;}; struct CVp{uint32_t Ci,Co,H,W,KH,KW,pad,st;};
struct ABp{uint32_t C,HW;}; struct AB2p{uint32_t n,C;}; struct T2p{uint32_t a,b;};
struct LNp{uint32_t tok,C;float e;}; struct MMp{uint32_t M,N,K,Nfull,colOffV;}; struct SHp{uint32_t seq,nh,d;};
struct ATp{uint32_t sQ,sK,d,H;float sc;}; struct N1p{uint32_t n;}; struct GGp{uint32_t tok,C2;};
static void matmul(Buf& a,Buf& b,Buf& y,uint32_t M,uint32_t N,uint32_t K,uint32_t Nfull=0,uint32_t colOffV=0);  // forward
static void silu(Buf& x,uint32_t n){ N1p p{n}; op(SILU,{&x,&x},&p,4,(n+255)/256,1,1); }
static void groupnorm(Buf& x,Buf& g,Buf& b,Buf& y,uint32_t Cc,uint32_t HW){ GNp p{Cc,HW,32,1e-5f}; op(GN,{&x,&g,&b,&y},&p,16,32,1,1); }
// переиспользуемый Col-буфер для im2col (один на весь forward, барьеры сериализуют доступ)
static Buf gCol; static VkDeviceSize gColCap=0;
static bool colFit(VkDeviceSize bytes){
    if (gColCap==0){ VkDeviceSize cap=(VkDeviceSize)832*9*64*64*4; if(bytes>cap)cap=bytes; gCol=C_->alloc(cap,true); gColCap=cap; }  // 832ch@64² ≈122МБ < лимит 128МБ
    return bytes<=gColCap;  // если внезапно больше выделенного — caller сделает direct
}
struct ICp{uint32_t Cin,Hin,Win,KH,KW,pad,stride,Wout,colOff,chunkN;};
// conv: 1×1 → matmul; K×K (любой stride) → im2col+matmul на нашем GEMM (с N-тайлингом если Col>лимита); иначе direct.
static void conv(Buf& x,Buf& w,Buf& y,uint32_t Ci,uint32_t Co,uint32_t H,uint32_t Wd,uint32_t K,uint32_t pad,uint32_t st){
    uint32_t Ho=(H+2*pad-K)/st+1, Wo=(Wd+2*pad-K)/st+1, Nout=Ho*Wo;
    if (st==1u && K==1u){ matmul(w,x,y,Co,H*Wd,Ci); return; }         // 1×1 conv = GEMM W[Co,Ci]·X[Ci,HW]
    uint32_t Kcol=Ci*K*K;
    if (colFit((VkDeviceSize)Kcol*Nout*4)){                           // целиком влезает: im2col + GEMM
        ICp p{Ci,H,Wd,K,K,pad,st,Wo,0u,Nout}; op(IM2COL,{&x,&gCol},&p,40,(Kcol*Nout+255)/256,1,1);
        matmul(w,gCol,y,Co,Nout,Kcol); return;
    }
    uint32_t chunk=(uint32_t)((gColCap/4u)/(VkDeviceSize)Kcol) & ~3u;  // N-тайлинг (мульт. 4)
    if (chunk>=4u){
        for (uint32_t off=0; off<Nout; off+=chunk){
            uint32_t cn=(off+chunk<=Nout)?chunk:(Nout-off);
            ICp p{Ci,H,Wd,K,K,pad,st,Wo,off,cn}; op(IM2COL,{&x,&gCol},&p,40,(Kcol*cn+255)/256,1,1);
            matmul(w,gCol,y,Co,cn,Kcol,Nout,off/4u);
        }
        return;
    }
    CVp p{Ci,Co,H,Wd,K,K,pad,st}; op(CONV,{&x,&w,&y},&p,32,(Wo+15)/16,(Ho+15)/16,Co); }  // fallback direct
static void addbias_c(Buf& y,Buf& b,uint32_t Cc,uint32_t HW){ ABp p{Cc,HW}; op(AB,{&y,&b},&p,8,(Cc*HW+255)/256,1,1); }
static void addbias_l(Buf& y,Buf& b,uint32_t n,uint32_t Cc){ AB2p p{n,Cc}; op(AB2,{&y,&b},&p,8,(n+255)/256,1,1); }
static void addv(Buf& a,Buf& b,Buf& y,uint32_t n){ N1p p{n}; op(ADD,{&a,&b,&y},&p,4,(n+255)/256,1,1); }
static void matmul(Buf& a,Buf& b,Buf& y,uint32_t M,uint32_t N,uint32_t K,uint32_t Nfull,uint32_t colOffV){
    if(Nfull==0u) Nfull=N; MMp p{M,N,K,Nfull,colOffV}; op(MM,{&a,&b,&y},&p,20,(N+127)/128,(M+127)/128,1); }

static int gDbg=0;
static void stat(Buf& b, uint32_t n, const char* lbl){
    if (gDbg>6) return;
    std::vector<float> v(n); C_->download(b,v.data(),(VkDeviceSize)n*4);
    float mx=0,s=0; int nan=0; for(uint32_t i=0;i<n;i++){ float x=v[i]; if(x!=x)nan++; if(fabsf(x)>mx)mx=fabsf(x); s+=x; }
    LOG("STAT %s n=%u absmax=%.3f mean=%.4f nan=%d",lbl,n,mx,s/n,nan);
}
// ResNet-блок: x[Cin,H,W] + temb[1280] → out[Cout,H,W]
static Buf resnet(const std::string& pf, Buf& x, Buf& temb, uint32_t Cin, uint32_t Cout, uint32_t H, uint32_t Wd){
    uint32_t HWi=H*Wd, Ni=Cin*HWi, No=Cout*HWi;
    Buf h=mk((VkDeviceSize)Ni*4); groupnorm(x,W(pf+".norm1.weight"),W(pf+".norm1.bias"),h,Cin,HWi); silu(h,Ni);
    Buf hc=mk((VkDeviceSize)No*4); conv(h,W(pf+".conv1.weight"),hc,Cin,Cout,H,Wd,3,1,1); addbias_c(hc,W(pf+".conv1.bias"),Cout,HWi);
    // time emb
    Buf ts=mk(1280*4); { N1p p{1280}; op(SILU,{&temb,&ts},&p,4,(1280+255)/256,1,1); }
    Buf tp=mk((VkDeviceSize)Cout*4); matmul(ts,W(pf+".time_emb_proj.weight"),tp,1,Cout,1280); addbias_l(tp,W(pf+".time_emb_proj.bias"),Cout,Cout);
    { ABp p{Cout,HWi}; op(AB,{&hc,&tp},&p,8,(No+255)/256,1,1); }   // hc += tp broadcast
    Buf h2=mk((VkDeviceSize)No*4); groupnorm(hc,W(pf+".norm2.weight"),W(pf+".norm2.bias"),h2,Cout,HWi); silu(h2,No);
    Buf hc2=mk((VkDeviceSize)No*4); conv(h2,W(pf+".conv2.weight"),hc2,Cout,Cout,H,Wd,3,1,1); addbias_c(hc2,W(pf+".conv2.bias"),Cout,HWi);
    Buf out=mk((VkDeviceSize)No*4);
    if (Cin!=Cout){ Buf sx=mk((VkDeviceSize)No*4); conv(x,W(pf+".conv_shortcut.weight"),sx,Cin,Cout,H,Wd,1,0,1); addbias_c(sx,W(pf+".conv_shortcut.bias"),Cout,HWi); addv(sx,hc2,out,No); }
    else addv(x,hc2,out,No);
    return out;
}

// Transformer-блок (SpatialTransformer): x[C,H,W] + ctx[77,768] → out[C,H,W]
static Buf transformer(const std::string& pf, Buf& x, Buf& ctx, uint32_t Cc, uint32_t H, uint32_t Wd, uint32_t nh){
    uint32_t HW=H*Wd, N=Cc*HW, d=Cc/nh, ctxN=77, C2=Cc*4, PROJ=Cc*8;
    Buf g1=mk((VkDeviceSize)N*4); groupnorm(x,W(pf+".norm.weight"),W(pf+".norm.bias"),g1,Cc,HW);
    Buf p1=mk((VkDeviceSize)N*4); conv(g1,W(pf+".proj_in.weight"),p1,Cc,Cc,H,Wd,1,0,1); addbias_c(p1,W(pf+".proj_in.bias"),Cc,HW);
    Buf t=mk((VkDeviceSize)N*4); { T2p p{Cc,HW}; op(T_CH,{&p1,&t},&p,8,(N+255)/256,1,1); }
    std::string b=pf+".transformer_blocks.0";
    Buf h=mk((VkDeviceSize)N*4),q=mk((VkDeviceSize)N*4),k=mk((VkDeviceSize)N*4),v=mk((VkDeviceSize)N*4),a=mk((VkDeviceSize)N*4);
    Buf qh=mk((VkDeviceSize)N*4),kh=mk((VkDeviceSize)N*4),vh=mk((VkDeviceSize)N*4),ah=mk((VkDeviceSize)N*4);
    auto split=[&](Buf& src,Buf& dst,uint32_t seq){ SHp p{seq,nh,d}; op(SPLIT,{&src,&dst},&p,12,(seq*nh*d+255)/256,1,1); };
    auto merge=[&](Buf& src,Buf& dst,uint32_t seq){ SHp p{seq,nh,d}; op(MERGE,{&src,&dst},&p,12,(seq*nh*d+255)/256,1,1); };
    // self-attn
    { LNp p{HW,Cc,1e-5f}; op(LN,{&t,&W(b+".norm1.weight"),&W(b+".norm1.bias"),&h},&p,12,HW,1,1); }
    matmul(h,W(b+".attn1.to_q.weight"),q,HW,Cc,Cc); matmul(h,W(b+".attn1.to_k.weight"),k,HW,Cc,Cc); matmul(h,W(b+".attn1.to_v.weight"),v,HW,Cc,Cc);
    split(q,qh,HW); split(k,kh,HW); split(v,vh,HW);
    { ATp p{HW,HW,d,nh,1.0f/sqrtf((float)d)}; if(d<=40){op(ATTN,{&qh,&kh,&vh,&ah},&p,20,(HW+127)/128,nh,1);}else{op(ATTN_BIG,{&qh,&kh,&vh,&ah},&p,20,(HW+127)/128,nh,1);} }
    merge(ah,a,HW); { Buf ao=mk((VkDeviceSize)N*4); matmul(a,W(b+".attn1.to_out.0.weight"),ao,HW,Cc,Cc); addbias_l(ao,W(b+".attn1.to_out.0.bias"),N,Cc); addv(t,ao,t,N); }
    // cross-attn
    { LNp p{HW,Cc,1e-5f}; op(LN,{&t,&W(b+".norm2.weight"),&W(b+".norm2.bias"),&h},&p,12,HW,1,1); }
    matmul(h,W(b+".attn2.to_q.weight"),q,HW,Cc,Cc);
    Buf kc=mk((VkDeviceSize)ctxN*Cc*4),vc=mk((VkDeviceSize)ctxN*Cc*4),khc=mk((VkDeviceSize)ctxN*Cc*4),vhc=mk((VkDeviceSize)ctxN*Cc*4);
    matmul(ctx,W(b+".attn2.to_k.weight"),kc,ctxN,Cc,768); matmul(ctx,W(b+".attn2.to_v.weight"),vc,ctxN,Cc,768);
    split(q,qh,HW); split(kc,khc,ctxN); split(vc,vhc,ctxN);
    { ATp p{HW,ctxN,d,nh,1.0f/sqrtf((float)d)}; if(d<=40){op(ATTN,{&qh,&khc,&vhc,&ah},&p,20,(HW+127)/128,nh,1);}else{op(ATTN_BIG,{&qh,&khc,&vhc,&ah},&p,20,(HW+127)/128,nh,1);} }
    merge(ah,a,HW); { Buf ao=mk((VkDeviceSize)N*4); matmul(a,W(b+".attn2.to_out.0.weight"),ao,HW,Cc,Cc); addbias_l(ao,W(b+".attn2.to_out.0.bias"),N,Cc); addv(t,ao,t,N); }
    // ff GEGLU
    { LNp p{HW,Cc,1e-5f}; op(LN,{&t,&W(b+".norm3.weight"),&W(b+".norm3.bias"),&h},&p,12,HW,1,1); }
    Buf gg=mk((VkDeviceSize)HW*PROJ*4); matmul(h,W(b+".ff.net.0.proj.weight"),gg,HW,PROJ,Cc); addbias_l(gg,W(b+".ff.net.0.proj.bias"),HW*PROJ,PROJ);
    Buf gf=mk((VkDeviceSize)HW*C2*4); { GGp p{HW,C2}; op(GEGLU,{&gg,&gf},&p,8,(HW*C2+255)/256,1,1); }
    Buf ff=mk((VkDeviceSize)N*4); matmul(gf,W(b+".ff.net.2.weight"),ff,HW,Cc,C2); addbias_l(ff,W(b+".ff.net.2.bias"),N,Cc); addv(t,ff,t,N);
    // proj_out + residual
    Buf tt=mk((VkDeviceSize)N*4); { T2p p{HW,Cc}; op(T_HC,{&t,&tt},&p,8,(N+255)/256,1,1); }
    Buf po=mk((VkDeviceSize)N*4); conv(tt,W(pf+".proj_out.weight"),po,Cc,Cc,H,Wd,1,0,1); addbias_c(po,W(pf+".proj_out.bias"),Cc,HW);
    Buf out=mk((VkDeviceSize)N*4); addv(x,po,out,N);
    return out;
}

// concat по каналам: [Cp,HW] ++ [Cs,HW] → [(Cp+Cs),HW] (смежная память)
static Buf concat(Buf& prev, uint32_t Cp, Buf& skip, uint32_t Cs, uint32_t HW){
    Buf out=mk((VkDeviceSize)(Cp+Cs)*HW*4);
    VkBufferCopy c1{0,0,(VkDeviceSize)Cp*HW*4}, c2{0,(VkDeviceSize)Cp*HW*4,(VkDeviceSize)Cs*HW*4};
    if (gProfile){ VkCommandBuffer cmd=C_->beginCmd(); vkCmdCopyBuffer(cmd,prev.buf,out.buf,1,&c1);
        vkCmdCopyBuffer(cmd,skip.buf,out.buf,1,&c2); C_->endCmd(cmd); return out; }
    barrier();  // дождаться записи prev/skip перед копированием
    vkCmdCopyBuffer(gCmd,prev.buf,out.buf,1,&c1);
    vkCmdCopyBuffer(gCmd,skip.buf,out.buf,1,&c2);
    barrier();  // копии видимы последующим шейдерам
    return out;
}
static Buf downsample(const std::string& pf, Buf& x, uint32_t Cc, uint32_t H, uint32_t Wd){
    uint32_t Ho=H/2, Wo=Wd/2; Buf y=mk((VkDeviceSize)Cc*Ho*Wo*4);
    conv(x,W(pf+".conv.weight"),y,Cc,Cc,H,Wd,3,1,2); addbias_c(y,W(pf+".conv.bias"),Cc,Ho*Wo); return y;
}
static Buf upsample(const std::string& pf, Buf& x, uint32_t Cc, uint32_t H, uint32_t Wd){
    uint32_t Ho=H*2, Wo=Wd*2; Buf u=mk((VkDeviceSize)Cc*Ho*Wo*4);  // 2× upsample (геометрия)
    { struct{uint32_t C,H,W;}p{Cc,H,Wd}; op(UP,{&x,&u},&p,12,(Cc*Ho*Wo+255)/256,1,1); }
    Buf y=mk((VkDeviceSize)Cc*Ho*Wo*4); conv(u,W(pf+".conv.weight"),y,Cc,Cc,Ho,Wo,3,1,1); addbias_c(y,W(pf+".conv.bias"),Cc,Ho*Wo); return y;
}
static double corrCheck(Buf& got, const std::string& refName, uint32_t n){
    std::string p=DIR_+"/"+refName+".bin"; int64_t sz=fsize(p); if(sz<0) return -1;
    std::vector<__fp16> ref(sz/2); FILE* f=fopen(p.c_str(),"rb"); fread(ref.data(),1,sz,f); fclose(f);
    std::vector<float> hy(n); C_->download(got,hy.data(),(VkDeviceSize)n*4);
    double se=0,sr=0; for(uint32_t i=0;i<n;i+=17){ se+=fabsf((float)ref[i]-hy[i]); sr+=fabsf((float)ref[i]); }
    return se/(sr+1e-6);
}
// Полный UNet forward: lat[4,64,64], tembProj[320], ctx[77,768] → noise[4,64,64].
// Использует persistent веса (WC_) и scratch (SCRATCH_).
static void resetKernelPools(){ for(int i=0;i<NSH;i++) if(gKi[i]) gK[i].resetPool(*C_); }
static Buf runGraph(Buf& lat, Buf& tembp, Buf& ctx){
    resetKernelPools();              // дескриптор-сеты прошлого forward освобождаем
    if(!gProfile) gCmd=C_->beginCmd();  // один command buffer на весь граф (в профиле — по-оп)
    // time embedding MLP
    Buf t1=mk(1280*4); matmul(tembp,W("time_embedding.linear_1.weight"),t1,1,1280,320); addbias_l(t1,W("time_embedding.linear_1.bias"),1280,1280); silu(t1,1280);
    Buf temb=mk(1280*4); matmul(t1,W("time_embedding.linear_2.weight"),temb,1,1280,1280); addbias_l(temb,W("time_embedding.linear_2.bias"),1280,1280);
    // conv_in: 4→320
    Buf x=mk((VkDeviceSize)320*64*64*4); conv(lat,W("conv_in.weight"),x,4,320,64,64,3,1,1); addbias_c(x,W("conv_in.bias"),320,64*64);
    std::vector<Buf> skips; skips.push_back(x);
    uint32_t bo[4]={320,640,1280,1280};
    // ---- DOWN ----
    uint32_t Cprev=320, H=64;
    for (int i=0;i<4;i++){
        uint32_t Cout=bo[i]; bool attn=(i<3);
        char pre[48];
        for (int r=0;r<2;r++){
            snprintf(pre,48,"down_blocks.%d.resnets.%d",i,r);
            Buf nx=resnet(pre,x,temb,(r==0)?Cprev:Cout,Cout,H,H); x=nx; Cprev=Cout;
            if (attn){ snprintf(pre,48,"down_blocks.%d.attentions.%d",i,r); x=transformer(pre,x,ctx,Cout,H,H,8); }
            skips.push_back(x);
        }
        if (i<3){ snprintf(pre,48,"down_blocks.%d.downsamplers.0",i); x=downsample(pre,x,Cout,H,H); H/=2; skips.push_back(x); }
    }
    LOG("GRAPH down done H=%d C=%d",H,Cprev);
    // ---- MID ---- (1280, 8×8): resnet+transformer+resnet
    x=resnet("mid_block.resnets.0",x,temb,1280,1280,H,H);
    x=transformer("mid_block.attentions.0",x,ctx,1280,H,H,8);
    x=resnet("mid_block.resnets.1",x,temb,1280,1280,H,H);
    LOG("GRAPH mid done");
    // ---- UP ---- up[i]: 3 resnet (cat skip), attn (кроме up0), upsample (кроме up3)
    uint32_t upout[4]={1280,1280,640,320};
    for (int i=0;i<4;i++){
        uint32_t Cout=upout[i]; bool attn=(i>0); bool up=(i<3);
        char pre[48];
        for (int r=0;r<3;r++){
            Buf skip=skips.back(); skips.pop_back();
            uint32_t Cs=bo[3-i]; // каналы skip соответствуют уровню
            // точные каналы skip берём из размера буфера
            uint32_t Cskip=(uint32_t)(skip.size/4/((VkDeviceSize)H*H));  // fp32: 4 байта/элемент
            uint32_t Cin=Cprev+Cskip;
            Buf cc=concat(x,Cprev,skip,Cskip,H*H);
            snprintf(pre,48,"up_blocks.%d.resnets.%d",i,r);
            x=resnet(pre,cc,temb,Cin,Cout,H,H); Cprev=Cout;
            if (attn){ snprintf(pre,48,"up_blocks.%d.attentions.%d",i,r); x=transformer(pre,x,ctx,Cout,H,H,8); }
        }
        if (up){ snprintf(pre,48,"up_blocks.%d.upsamplers.0",i); x=upsample(pre,x,Cout,H,H); H*=2; }
    }
    LOG("GRAPH up done H=%d",H);
    // ---- OUT ---- groupnorm+silu+conv_out(320→4)
    Buf gno=mk((VkDeviceSize)320*H*H*4); groupnorm(x,W("conv_norm_out.weight"),W("conv_norm_out.bias"),gno,320,H*H); silu(gno,320*H*H);
    Buf y=mk((VkDeviceSize)4*H*H*4); conv(gno,W("conv_out.weight"),y,320,4,H,H,3,1,1); addbias_c(y,W("conv_out.bias"),4,H*H);
    if (!gProfile){  // один submit на весь forward
        vkEndCommandBuffer(gCmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&gCmd;
        vkQueueSubmit(C_->queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(C_->queue);
        vkFreeCommandBuffers(C_->dev,C_->pool,1,&gCmd); gCmd=VK_NULL_HANDLE;
    }
    return y;
}
static VkCtx gCtx; static bool gInit=false;
static std::vector<std::vector<uint8_t>> gSh;
} // namespace unet

// init: persistent ctx + shaders + dir (веса lazy)
extern "C" JNIEXPORT jint JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_unetInit(
        JNIEnv* env, jobject, jobjectArray shaders, jstring dir) {
    using namespace unet;
    if (gInit) return 0;
    if(!gCtx.init()) return -1; C_=&gCtx;
    gSh.assign(NSH,{});
    for (int i=0;i<NSH;i++){ jbyteArray e=(jbyteArray)env->GetObjectArrayElement(shaders,i);
        jsize l=env->GetArrayLength(e); gSh[i].resize(l); env->GetByteArrayRegion(e,0,l,(jbyte*)gSh[i].data()); }
    SH_=&gSh;
    const char* cd=env->GetStringUTFChars(dir,nullptr); DIR_=cd; env->ReleaseStringUTFChars(dir,cd);
    gInit=true; LOG("UNET init OK"); return 0;
}

// self-test: эталонный вход IN_* из файлов → граф → corr vs OUT_y (диагностика fp32)
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_unetSelfTest(JNIEnv*, jobject) {
    using namespace unet; if(!gInit) return -2;
    auto loadF=[&](const std::string& nm, int n){ Buf b=mk((VkDeviceSize)n*4);
        std::string p=DIR_+"/"+nm+".bin"; FILE* f=fopen(p.c_str(),"rb"); std::vector<__fp16> h(n);
        if(f){ fread(h.data(),2,n,f); fclose(f); } std::vector<float> v(n); for(int i=0;i<n;i++) v[i]=(float)h[i];
        gCtx.upload(b,v.data(),(VkDeviceSize)n*4); return b; };
    Buf lat=loadF("IN_lat",4*64*64), ctx=loadF("IN_ctx",77*768), tp=loadF("IN_temb_proj",320);
    Buf noise=runGraph(lat,tp,ctx);
    double e=corrCheck(noise,"OUT_y",4*64*64);
    LOG("UNET SELFTEST fp32 corr=%.4f %s",e,e<0.1?"OK":"FAIL");
    for (auto& b: SCRATCH_) gCtx.free(b); SCRATCH_.clear();
    if (has("IN_cond")){
        Buf l2=loadF("IN_lat",4*64*64), t2=loadF("IN_temb999",320), cc=loadF("IN_cond",77*768);
        Buf ec=runGraph(l2,t2,cc); std::vector<float> vc(4*64*64); gCtx.download(ec,vc.data(),(VkDeviceSize)4*64*64*4);
        for(auto&b:SCRATCH_)gCtx.free(b); SCRATCH_.clear();
        Buf l3=loadF("IN_lat",4*64*64), t3=loadF("IN_temb999",320), uc=loadF("IN_unc",77*768);
        Buf eu=runGraph(l3,t3,uc); std::vector<float> vu(4*64*64); gCtx.download(eu,vu.data(),(VkDeviceSize)4*64*64*4);
        for(auto&b:SCRATCH_)gCtx.free(b); SCRATCH_.clear();
        double d=0; for(int i=0;i<4*64*64;i++) d+=fabsf(vc[i]-vu[i]); d/=(4*64*64);
        LOG("UNET CTX-SENS |eps_cond-eps_unc|=%.4f (PyTorch=0.0295)",d);
    }
    return e;
}

// профиль: один forward по-операционно, лог GPU-времени по категориям
extern "C" JNIEXPORT void JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_unetProfile(JNIEnv*, jobject) {
    using namespace unet; if(!gInit) return;
    auto loadF=[&](const std::string& nm, int n){ Buf b=mk((VkDeviceSize)n*4);
        std::string p=DIR_+"/"+nm+".bin"; FILE* f=fopen(p.c_str(),"rb"); std::vector<__fp16> h(n);
        if(f){ fread(h.data(),2,n,f); fclose(f); } std::vector<float> v(n); for(int i=0;i<n;i++) v[i]=(float)h[i];
        gCtx.upload(b,v.data(),(VkDeviceSize)n*4); return b; };
    for(int i=0;i<NSH;i++){ gProfT[i]=0; gProfN[i]=0; }
    Buf lat=loadF("IN_lat",4*64*64), ctx=loadF("IN_ctx",77*768), tp=loadF("IN_temb_proj",320);
    gProfile=true;
    Buf noise=runGraph(lat,tp,ctx); (void)noise;
    gProfile=false;
    const char* nm[NSH]={"GN","SILU","CONV","MM","AB","AB2","ADD","LN","SPLIT","ATTN","MERGE","GEGLU","T_CH","T_HC","UP","ATTN_BIG","IM2COL"};
    double tot=0; for(int i=0;i<NSH;i++) tot+=gProfT[i];
    LOG("=== PROFILE forward total=%.1f ms ===", tot*1000.0);
    for(int i=0;i<NSH;i++) if(gProfN[i]>0)
        LOG("PROF %-8s %7.1f ms  (%4.1f%%)  n=%d", nm[i], gProfT[i]*1000.0, 100.0*gProfT[i]/tot, gProfN[i]);
    for (auto& b: SCRATCH_) gCtx.free(b); SCRATCH_.clear();
}

// forward: latFp16[4*64*64*2], tembFp16[320*2], ctxFp16[77*768*2] → noiseFp16[4*64*64*2]
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_unetForward(
        JNIEnv* env, jobject, jbyteArray latB, jbyteArray tembB, jbyteArray ctxB) {
    using namespace unet; if(!gInit) return nullptr;
    auto up=[&](jbyteArray a, uint32_t n){ Buf b=mk((VkDeviceSize)n*4); jsize l=env->GetArrayLength(a);
        std::vector<uint8_t> v(l); env->GetByteArrayRegion(a,0,l,(jbyte*)v.data()); gCtx.upload(b,v.data(),v.size()); return b; };
    Buf lat=up(latB,4*64*64), temb=up(tembB,320), ctx=up(ctxB,77*768);
    Buf noise=runGraph(lat,temb,ctx);
    std::vector<uint8_t> out((size_t)4*64*64*4); gCtx.download(noise,out.data(),out.size());
    jbyteArray res=env->NewByteArray((jsize)out.size()); env->SetByteArrayRegion(res,0,(jsize)out.size(),(jbyte*)out.data());
    for (auto& b: SCRATCH_) gCtx.free(b); SCRATCH_.clear();   // освобождаем временные, веса остаются
    return res;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_unetRelease(JNIEnv*, jobject) {
    using namespace unet; if(!gInit) return;
    for (int i=0;i<NSH;i++) if(gKi[i]){ gK[i].destroy(gCtx); gKi[i]=false; }
    if (gColCap){ gCtx.free(gCol); gColCap=0; }
    for (auto& kv: WC_) gCtx.free(kv.second); WC_.clear();
    gCtx.destroy(); gInit=false;
}

// ====================== JNI: TRANSFORMER BLOCK ======================
// shaders[13]: groupnorm,conv2d,addbias,addbias2,t_chw2hwc,layernorm,matmul,split_heads,attention,merge_heads,add,t_hwc2chw,geglu
// weights[29]: x,ctx,gn_w,gn_b,pin_w,pin_b,n1_w,n1_b,q1,k1,v1,o1,o1_b,n2_w,n2_b,q2,k2,v2,o2,o2_b,n3_w,n3_b,geglu,geglu_b,ffout,ffout_b,pout_w,pout_b,y
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_runTransformerBlock(
        JNIEnv* env, jobject, jobjectArray shaders, jobjectArray weights,
        jint C, jint H, jint W, jint ctxN, jint ctxD, jint nh) {
    VkCtx c; if(!c.init()) return -1;
    auto getArr=[&](jobjectArray a,int i){ jbyteArray e=(jbyteArray)env->GetObjectArrayElement(a,i);
        jsize l=env->GetArrayLength(e); std::vector<uint8_t> v(l); env->GetByteArrayRegion(e,0,l,(jbyte*)v.data()); return v; };
    std::vector<std::vector<uint8_t>> sh(13), w(29);
    for(int i=0;i<13;i++) sh[i]=getArr(shaders,i);
    for(int i=0;i<29;i++) w[i]=getArr(weights,i);

    uint32_t HW=(uint32_t)H*W, d=(uint32_t)C/nh, C2=1280, PROJ=2560;
    auto mk=[&](int i, VkDeviceSize bytes){ Buf b=c.alloc(bytes,true); c.upload(b,w[i].data(),bytes); return b; };
    // веса
    Buf bx=mk(0,(VkDeviceSize)C*HW*2), bctx=mk(1,(VkDeviceSize)ctxN*ctxD*2);
    Buf gnw=mk(2,C*2),gnb=mk(3,C*2), pinw=mk(4,(VkDeviceSize)C*C*2),pinb=mk(5,C*2);
    Buf n1w=mk(6,C*2),n1b=mk(7,C*2), q1=mk(8,(VkDeviceSize)C*C*2),k1=mk(9,(VkDeviceSize)C*C*2),v1=mk(10,(VkDeviceSize)C*C*2),o1=mk(11,(VkDeviceSize)C*C*2),o1b=mk(12,C*2);
    Buf n2w=mk(13,C*2),n2b=mk(14,C*2), q2=mk(15,(VkDeviceSize)C*C*2),k2=mk(16,(VkDeviceSize)ctxD*C*2),v2=mk(17,(VkDeviceSize)ctxD*C*2),o2=mk(18,(VkDeviceSize)C*C*2),o2b=mk(19,C*2);
    Buf n3w=mk(20,C*2),n3b=mk(21,C*2), ggw=mk(22,(VkDeviceSize)C*PROJ*2),ggb=mk(23,(VkDeviceSize)PROJ*2),fow=mk(24,(VkDeviceSize)C2*C*2),fob=mk(25,C*2);
    Buf poutw=mk(26,(VkDeviceSize)C*C*2),poutb=mk(27,C*2);
    std::vector<__fp16> yref((size_t)C*HW); memcpy(yref.data(),w[28].data(),(size_t)C*HW*2);

    auto A=[&](VkDeviceSize n){ return c.alloc(n*2,true); };
    Buf g1=A(C*HW),p1=A(C*HW),t=A(HW*C),h=A(HW*C),q=A(HW*C),k=A(HW*C),v=A(HW*C),a=A(HW*C);
    Buf qh=A((VkDeviceSize)nh*HW*d),kh=A((VkDeviceSize)nh*HW*d),vh=A((VkDeviceSize)nh*HW*d),ah=A((VkDeviceSize)nh*HW*d);
    Buf kc=A((VkDeviceSize)ctxN*C),vc=A((VkDeviceSize)ctxN*C),khc=A((VkDeviceSize)nh*ctxN*d),vhc=A((VkDeviceSize)nh*ctxN*d);
    Buf gg=A((VkDeviceSize)HW*PROJ),gf=A((VkDeviceSize)HW*C2),ff=A(HW*C),tt=A(C*HW),po=A(C*HW),out=A(C*HW);

    auto op=[&](int si,std::vector<Buf*> bufs,const void* push,uint32_t pb,uint32_t gx,uint32_t gy,uint32_t gz){
        Kernel kr; kr.create(c,sh[si].data(),sh[si].size(),(int)bufs.size(),pb);
        VkDescriptorSet ds=kr.makeSet(c,bufs); VkCommandBuffer cmd=c.beginCmd();
        kr.record(cmd,ds,push,gx,gy,gz); c.endCmd(cmd); kr.destroy(c); };
    // push-структуры
    struct GN{uint32_t C,HW,G;float e;}; struct CV{uint32_t Ci,Co,H,W,KH,KW,pad,stride;};
    struct AB{uint32_t C,HW;}; struct AB2{uint32_t n,C;}; struct T2{uint32_t a,b;};
    struct LN{uint32_t tok,C;float e;}; struct MM{uint32_t M,N,K;}; struct SH{uint32_t seq,nh,d;};
    struct AT{uint32_t sQ,sK,d,H;float sc;}; struct N1{uint32_t n;}; struct GG{uint32_t tok,C2;};
    uint32_t g_chw=(C*HW+255)/256, g_tok=(HW*C+255)/256;
    auto MMrun=[&](Buf* AA,Buf* BB,Buf* CC,uint32_t M,uint32_t N,uint32_t K){ MM m{M,N,K}; op(6,{AA,BB,CC},&m,12,(N+127)/128,(M+127)/128,1); };

    // --- граф ---
    { GN gn{(uint32_t)C,HW,32,1e-5f}; op(0,{&bx,&gnw,&gnb,&g1},&gn,16,32,1,1); }     // groupnorm
    { CV cv{(uint32_t)C,(uint32_t)C,(uint32_t)H,(uint32_t)W,1,1,0,1}; op(1,{&g1,&pinw,&p1},&cv,32,((uint32_t)W+15)/16,((uint32_t)H+15)/16,(uint32_t)C); } // proj_in conv1x1
    { AB ab{(uint32_t)C,HW}; op(2,{&p1,&pinb},&ab,8,g_chw,1,1); }
    { T2 tr{(uint32_t)C,HW}; op(4,{&p1,&t},&tr,8,g_chw,1,1); }                        // [C,HW]→[HW,C]
    // self-attn
    { LN ln{HW,(uint32_t)C,1e-5f}; op(5,{&t,&n1w,&n1b,&h},&ln,12,HW,1,1); }
    MMrun(&h,&q1,&q,HW,(uint32_t)C,(uint32_t)C); MMrun(&h,&k1,&k,HW,(uint32_t)C,(uint32_t)C); MMrun(&h,&v1,&v,HW,(uint32_t)C,(uint32_t)C);
    { SH s{HW,(uint32_t)nh,d}; op(7,{&q,&qh},&s,12,(nh*HW*d+255)/256,1,1); op(7,{&k,&kh},&s,12,(nh*HW*d+255)/256,1,1); op(7,{&v,&vh},&s,12,(nh*HW*d+255)/256,1,1); }
    { AT at{HW,HW,d,(uint32_t)nh,1.0f/sqrtf((float)d)}; op(8,{&qh,&kh,&vh,&ah},&at,20,(HW+63)/64,(uint32_t)nh,1); }
    { SH s{HW,(uint32_t)nh,d}; op(9,{&ah,&a},&s,12,(nh*HW*d+255)/256,1,1); }          // merge
    { Buf ao=A(HW*C); MMrun(&a,&o1,&ao,HW,(uint32_t)C,(uint32_t)C); AB2 ab{HW*(uint32_t)C,(uint32_t)C}; op(3,{&ao,&o1b},&ab,8,g_tok,1,1);
      N1 n{HW*(uint32_t)C}; op(10,{&t,&ao,&t},&n,4,g_tok,1,1); c.free(ao); }          // t += to_out(attn)
    // cross-attn
    { LN ln{HW,(uint32_t)C,1e-5f}; op(5,{&t,&n2w,&n2b,&h},&ln,12,HW,1,1); }
    MMrun(&h,&q2,&q,HW,(uint32_t)C,(uint32_t)C);
    MMrun(&bctx,&k2,&kc,(uint32_t)ctxN,(uint32_t)C,(uint32_t)ctxD); MMrun(&bctx,&v2,&vc,(uint32_t)ctxN,(uint32_t)C,(uint32_t)ctxD);
    { SH s{HW,(uint32_t)nh,d}; op(7,{&q,&qh},&s,12,(nh*HW*d+255)/256,1,1); }
    { SH s{(uint32_t)ctxN,(uint32_t)nh,d}; op(7,{&kc,&khc},&s,12,(nh*ctxN*d+255)/256,1,1); op(7,{&vc,&vhc},&s,12,(nh*ctxN*d+255)/256,1,1); }
    { AT at{HW,(uint32_t)ctxN,d,(uint32_t)nh,1.0f/sqrtf((float)d)}; op(8,{&qh,&khc,&vhc,&ah},&at,20,(HW+63)/64,(uint32_t)nh,1); }
    { SH s{HW,(uint32_t)nh,d}; op(9,{&ah,&a},&s,12,(nh*HW*d+255)/256,1,1); }
    { Buf ao=A(HW*C); MMrun(&a,&o2,&ao,HW,(uint32_t)C,(uint32_t)C); AB2 ab{HW*(uint32_t)C,(uint32_t)C}; op(3,{&ao,&o2b},&ab,8,g_tok,1,1);
      N1 n{HW*(uint32_t)C}; op(10,{&t,&ao,&t},&n,4,g_tok,1,1); c.free(ao); }
    // ff (GEGLU)
    { LN ln{HW,(uint32_t)C,1e-5f}; op(5,{&t,&n3w,&n3b,&h},&ln,12,HW,1,1); }
    MMrun(&h,&ggw,&gg,HW,PROJ,(uint32_t)C);
    { AB2 ab{HW*PROJ,PROJ}; op(3,{&gg,&ggb},&ab,8,(HW*PROJ+255)/256,1,1); }
    { GG gge{HW,C2}; op(12,{&gg,&gf},&gge,8,(HW*C2+255)/256,1,1); }                   // geglu combine
    MMrun(&gf,&fow,&ff,HW,(uint32_t)C,C2);
    { AB2 ab{HW*(uint32_t)C,(uint32_t)C}; op(3,{&ff,&fob},&ab,8,g_tok,1,1);
      N1 n{HW*(uint32_t)C}; op(10,{&t,&ff,&t},&n,4,g_tok,1,1); }
    // proj_out + residual
    { T2 tr{HW,(uint32_t)C}; op(11,{&t,&tt},&tr,8,g_tok,1,1); }                       // [HW,C]→[C,HW]
    { CV cv{(uint32_t)C,(uint32_t)C,(uint32_t)H,(uint32_t)W,1,1,0,1}; op(1,{&tt,&poutw,&po},&cv,32,((uint32_t)W+15)/16,((uint32_t)H+15)/16,(uint32_t)C); }
    { AB ab{(uint32_t)C,HW}; op(2,{&po,&poutb},&ab,8,g_chw,1,1); }
    { N1 n{(uint32_t)C*HW}; op(10,{&bx,&po,&out},&n,4,g_chw,1,1); }

    std::vector<__fp16> hy((size_t)C*HW); c.download(out,hy.data(),(VkDeviceSize)C*HW*2);
    double se=0,sr=0; for(uint32_t i=0;i<(uint32_t)C*HW;i+=53){ se+=fabsf((float)yref[i]-(float)hy[i]); sr+=fabsf((float)yref[i]); }
    double e=se/(sr+1e-6); LOG("TRANSFORMER C%d %dx%d corr=%.4f %s",C,H,W,e,e<0.06?"OK(fp16)":"FAIL");
    c.destroy(); return e;
}

// ====================== JNI: ATTENTION ======================
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchAttention(
        JNIEnv* env, jobject, jbyteArray spirv, jint H, jint seqQ, jint seqK, jint d, jint tq, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    jsize l=env->GetArrayLength(spirv); std::vector<uint8_t> spv(l); env->GetByteArrayRegion(spirv,0,l,(jbyte*)spv.data());
    VkDeviceSize szQ=(VkDeviceSize)H*seqQ*d*4, szK=(VkDeviceSize)H*seqK*d*4, szO=szQ;
    Buf bQ=c.alloc(szQ,true), bK=c.alloc(szK,true), bV=c.alloc(szK,true), bO=c.alloc(szO,true);
    std::vector<float> hQ((size_t)H*seqQ*d), hK((size_t)H*seqK*d), hV((size_t)H*seqK*d);
    for(size_t i=0;i<hQ.size();i++) hQ[i]=(((i*131u+7u)%23u)*0.05f-0.55f);
    for(size_t i=0;i<hK.size();i++) hK[i]=(((i*61u+13u)%19u)*0.05f-0.45f);
    for(size_t i=0;i<hV.size();i++) hV[i]=(((i*97u+5u)%17u)*0.05f-0.4f);
    c.upload(bQ,hQ.data(),szQ); c.upload(bK,hK.data(),szK); c.upload(bV,hV.data(),szK);

    Kernel k; k.create(c,spv.data(),spv.size(),4,20); VkDescriptorSet ds=k.makeSet(c,{&bQ,&bK,&bV,&bO});
    float scale=1.0f/sqrtf((float)d);
    struct { uint32_t sQ,sK,d,H; float sc; } pc{(uint32_t)seqQ,(uint32_t)seqK,(uint32_t)d,(uint32_t)H,scale};
    uint32_t gx=((uint32_t)seqQ+(uint32_t)tq-1u)/(uint32_t)tq, gy=(uint32_t)H;

    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,&pc,gx,gy,1); c.endCmd(cmd); }
    // верификация: эталонный softmax-attention на CPU для нескольких (h,i,k)
    { std::vector<float> hO((size_t)H*seqQ*d); c.download(bO,hO.data(),szO);
      double se=0,sr=0;
      for (int t=0;t<8;t++){
          int h=(t)%H, i=(t*53+1)%seqQ, kk=(t*29+2)%d;
          std::vector<float> sc(seqK); float mx=-1e30f;
          for (int j=0;j<seqK;j++){ float s=0; for(int q=0;q<d;q++) s+=(float)hQ[((size_t)h*seqQ+i)*d+q]*(float)hK[((size_t)h*seqK+j)*d+q];
              s*=scale; sc[j]=s; if(s>mx)mx=s; }
          float sum=0; for(int j=0;j<seqK;j++){ sc[j]=expf(sc[j]-mx); sum+=sc[j]; }
          float ref=0; for(int j=0;j<seqK;j++) ref += sc[j]/sum*(float)hV[((size_t)h*seqK+j)*d+kk];
          se+=fabsf(ref-(float)hO[((size_t)h*seqQ+i)*d+kk]); sr+=fabsf(ref);
      }
      double e=se/(sr+1e-6); LOG("ATTN H%d sQ%d sK%d d%d corr=%.4f %s",H,seqQ,seqK,d,e,e<0.03?"OK":"FAIL"); }

    double sec=timeDispatch(c,k,ds,&pc,gx,gy,1,iters);
    // FLOPS attention ~ 2*H*seqQ*seqK*d (QK) + 2*H*seqQ*seqK*d (AV)
    double flops=4.0*(double)H*seqQ*seqK*d;
    LOG("attn H%d sQ%d sK%d d%d: %.3f ms, %.1f GFLOPS",H,seqQ,seqK,d,sec*1000,flops/sec/1e9);
    k.destroy(c); c.free(bQ);c.free(bK);c.free(bV);c.free(bO); c.destroy(); return sec*1000;
}

// ====================== JNI: CONV через im2col + GEMM ======================
// Принимает 2 SPIR-V: im2col и matmul. conv → Col[K,Ncol] → Y=W×Col.
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchConvGemm(
        JNIEnv* env, jobject, jbyteArray im2colSpv, jbyteArray matmulSpv,
        jint Cin, jint Cout, jint H, jint W, jint KH, jint KW, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    auto getSpv=[&](jbyteArray a){ jsize l=env->GetArrayLength(a); std::vector<uint8_t> v(l);
        env->GetByteArrayRegion(a,0,l,(jbyte*)v.data()); return v; };
    auto icSpv=getSpv(im2colSpv); auto mmSpv=getSpv(matmulSpv);
    int pad=KH/2;
    uint32_t Ncol=(uint32_t)H*W, Kcol=(uint32_t)Cin*KH*KW, Mc=(uint32_t)Cout;

    VkDeviceSize szX=(VkDeviceSize)Cin*H*W*4, szW=(VkDeviceSize)Cout*Kcol*4;
    VkDeviceSize szCol=(VkDeviceSize)Kcol*Ncol*4, szY=(VkDeviceSize)Cout*Ncol*4;
    Buf bX=c.alloc(szX,true), bW=c.alloc(szW,true), bCol=c.alloc(szCol,true), bY=c.alloc(szY,true);
    std::vector<float> hX((size_t)Cin*H*W), hW((size_t)Cout*Kcol);
    for (size_t i=0;i<hX.size();i++) hX[i]=(((i*131u+7u)%17u)*0.02f-0.16f);
    for (size_t i=0;i<hW.size();i++) hW[i]=(((i*61u+13u)%19u)*0.02f-0.18f);
    c.upload(bX,hX.data(),szX); c.upload(bW,hW.data(),szW);

    Kernel ic; ic.create(c,icSpv.data(),icSpv.size(),2,40);
    VkDescriptorSet icDs=ic.makeSet(c,{&bX,&bCol});
    uint32_t icPc[10]={(uint32_t)Cin,(uint32_t)H,(uint32_t)W,(uint32_t)KH,(uint32_t)KW,(uint32_t)pad,1u,(uint32_t)W,0u,Ncol};
    uint32_t icG=(Kcol*Ncol+255)/256;

    Kernel mm; mm.create(c,mmSpv.data(),mmSpv.size(),3,20);
    VkDescriptorSet mmDs=mm.makeSet(c,{&bW,&bCol,&bY});  // A=W[M,K], B=Col[K,N], C=Y[M,N]
    uint32_t mmPc[5]={Mc,Ncol,Kcol,Ncol,0u};
    uint32_t gx=(Ncol+127)/128, gy=(Mc+127)/128;

    // один проход conv = im2col → matmul (с барьером между)
    auto runConv=[&](VkCommandBuffer cmd){
        ic.record(cmd,icDs,icPc,icG,1,1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
        mm.record(cmd,mmDs,mmPc,gx,gy,1);
    };
    auto one=[&](){ VkCommandBuffer cmd=c.beginCmd(); runConv(cmd); c.endCmd(cmd); };
    one(); // прогрев + для верификации
    // верификация Y vs CPU
    { std::vector<float> hY((size_t)Cout*Ncol); c.download(bY,hY.data(),szY);
      double se=0,sr=0;
      for (int s=0;s<16;s++){ int co=(s*7+1)%Cout, oy=(s*13+2)%H, ox=(s*5+3)%W; float ref=0;
        for (int ci=0;ci<Cin;ci++) for(int ky=0;ky<KH;ky++) for(int kx=0;kx<KW;kx++){
            int iy=oy+ky-pad, ix=ox+kx-pad; if(iy<0||iy>=H||ix<0||ix>=W) continue;
            ref += (float)hX[((size_t)ci*H+iy)*W+ix]*(float)hW[(((size_t)co*Cin+ci)*KH+ky)*KW+kx]; }
        se+=fabsf(ref-(float)hY[(size_t)co*Ncol+(size_t)oy*W+ox]); sr+=fabsf(ref); }
      double e=se/(sr+1e-6);
      LOG("CONVGEMM Cin%d Cout%d %dx%d k%d corr=%.4f %s",Cin,Cout,H,W,KH,e,e<0.03?"OK":"FAIL"); }

    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<iters;i++) one();
    auto t1=std::chrono::high_resolution_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count()/iters;
    double flops=2.0*(double)Cout*Cin*H*W*KH*KW;
    LOG("convGEMM Cin%d Cout%d %dx%d k%d: %.3f ms, %.1f GFLOPS",Cin,Cout,H,W,KH,sec*1000,flops/sec/1e9);
    ic.destroy(c); mm.destroy(c); c.free(bX);c.free(bW);c.free(bCol);c.free(bY); c.destroy();
    return flops/sec/1e9;
}

// ====================== JNI: CONV2D ======================
// X[Cin,H,W] * W[Cout,Cin,KH,KW] -> Y[Cout,H,W], same padding, stride 1, N=1. fp16 storage, fp32 acc.
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchConv(
        JNIEnv* env, jobject, jbyteArray spirv, jint Cin, jint Cout, jint H, jint W, jint KH, jint KW, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    jsize spvLen=env->GetArrayLength(spirv); std::vector<uint8_t> spv(spvLen);
    env->GetByteArrayRegion(spirv,0,spvLen,(jbyte*)spv.data());
    int pad=KH/2;

    VkDeviceSize szX=(VkDeviceSize)Cin*H*W*4, szW=(VkDeviceSize)Cout*Cin*KH*KW*4, szY=(VkDeviceSize)Cout*H*W*4;
    Buf bX=c.alloc(szX,true), bW=c.alloc(szW,true), bY=c.alloc(szY,true);
    std::vector<float> hX((size_t)Cin*H*W), hW((size_t)Cout*Cin*KH*KW);
    for (size_t i=0;i<hX.size();i++) hX[i]=(((i*131u+7u)%17u)*0.02f-0.16f);
    for (size_t i=0;i<hW.size();i++) hW[i]=(((i*61u+13u)%19u)*0.02f-0.18f);
    c.upload(bX,hX.data(),szX); c.upload(bW,hW.data(),szW);

    Kernel k; k.create(c,spv.data(),spvLen,3,32);  // push: Cin,Cout,H,W,KH,KW,pad,stride
    VkDescriptorSet ds=k.makeSet(c,{&bX,&bW,&bY});
    uint32_t pc[8]={(uint32_t)Cin,(uint32_t)Cout,(uint32_t)H,(uint32_t)W,(uint32_t)KH,(uint32_t)KW,(uint32_t)pad,1u};
    uint32_t gx=((uint32_t)W+15)/16, gy=((uint32_t)H+15)/16, gz=(uint32_t)Cout;

    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,pc,gx,gy,gz); c.endCmd(cmd); }
    { std::vector<float> hY((size_t)Cout*H*W); c.download(bY,hY.data(),szY);
      double se=0,sr=0;
      for (int s=0;s<16;s++){
          int co=(s*7+1)%Cout, oy=(s*13+2)%H, ox=(s*5+3)%W;
          float ref=0;
          for (int ci=0;ci<Cin;ci++) for(int ky=0;ky<KH;ky++) for(int kx=0;kx<KW;kx++){
              int iy=oy+ky-pad, ix=ox+kx-pad;
              if (iy<0||iy>=H||ix<0||ix>=W) continue;
              ref += (float)hX[((size_t)ci*H+iy)*W+ix]*(float)hW[(((size_t)co*Cin+ci)*KH+ky)*KW+kx];
          }
          se+=fabsf(ref-(float)hY[((size_t)co*H+oy)*W+ox]); sr+=fabsf(ref);
      }
      double e=se/(sr+1e-6);
      LOG("CONV2D Cin%d Cout%d %dx%d k%dx%d corr=%.4f %s",Cin,Cout,H,W,KH,KW,e,e<0.02?"OK":"FAIL");
    }
    double sec=timeDispatch(c,k,ds,pc,gx,gy,gz,iters);
    double flops=2.0*(double)Cout*Cin*H*W*KH*KW;
    LOG("conv Cin%d Cout%d %dx%d k%d: %.3f ms, %.1f GFLOPS",Cin,Cout,H,W,KH,sec*1000,flops/sec/1e9);
    k.destroy(c); c.free(bX);c.free(bW);c.free(bY); c.destroy();
    return flops/sec/1e9;
}
