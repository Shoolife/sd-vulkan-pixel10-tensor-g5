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
        VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(phys,&pr); LOG("GPU: %s", pr.deviceName);
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

    void create(VkCtx& c, const void* spv, size_t spvLen, int numBuffers, uint32_t pushBytes){
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
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(uint32_t)numBuffers};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets=1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; vkCreateDescriptorPool(c.dev,&dpi,nullptr,&dp);
    }
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

    VkDeviceSize szA=(VkDeviceSize)M*K*2, szB=(VkDeviceSize)K*N*2, szC=(VkDeviceSize)M*N*2;
    Buf bA=c.alloc(szA,true), bB=c.alloc(szB,true), bC=c.alloc(szC,true);
    std::vector<__fp16> hA((size_t)M*K), hB((size_t)K*N);
    for (size_t i=0;i<hA.size();i++) hA[i]=(__fp16)(((i*131u+7u)%17u)*0.01f-0.08f);
    for (size_t i=0;i<hB.size();i++) hB[i]=(__fp16)(((i*61u+13u)%19u)*0.01f-0.09f);
    c.upload(bA,hA.data(),szA); c.upload(bB,hB.data(),szB);

    Kernel k; k.create(c,spv.data(),spvLen,3,12);
    VkDescriptorSet ds=k.makeSet(c,{&bA,&bB,&bC});
    uint32_t pc[3]={(uint32_t)M,(uint32_t)N,(uint32_t)K};
    uint32_t gx=((uint32_t)N+127)/128, gy=((uint32_t)M+127)/128;

    // прогрев + верификация
    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,pc,gx,gy,1); c.endCmd(cmd); }
    { std::vector<__fp16> hC((size_t)M*N); c.download(bC,hC.data(),szC);
      double se=0,sr=0; for(int s=0;s<16;s++){ int r=(s*97+3)%M,col=(s*53+11)%N; float ref=0;
        for(int kk=0;kk<K;kk++) ref+=(float)hA[(size_t)r*K+kk]*(float)hB[(size_t)kk*N+col];
        se+=fabsf(ref-(float)hC[(size_t)r*N+col]); sr+=fabsf(ref);} double e=se/(sr+1e-6);
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
    struct GN{uint32_t C,HW,G;float eps;}; struct CV{uint32_t Ci,Co,H,W,KH,KW,pad;};
    struct MM{uint32_t M,N,K;}; struct AB{uint32_t C,HW;}; struct N1{uint32_t n;};
    uint32_t gSilu=(N+255)/256, gAB=(N+255)/256;
    GN gn{(uint32_t)C,HW,G,1e-5f}; CV cv{(uint32_t)C,(uint32_t)C,(uint32_t)H,(uint32_t)W,3,3,1};
    uint32_t cgx=((uint32_t)W+15)/16, cgy=((uint32_t)H+15)/16, cgz=(uint32_t)C;

    // --- граф ResNet ---
    op(0,{&bx,&n1w,&n1b,&h},&gn,16,G,1,1);              // h = groupnorm(x)
    { N1 n{N}; op(1,{&h,&h},&n,4,gSilu,1,1); }          // h = silu(h)  (in-place)
    op(2,{&h,&c1w,&hc},&cv,28,cgx,cgy,cgz);             // hc = conv1(h)
    { AB ab{(uint32_t)C,HW}; op(4,{&hc,&c1b},&ab,8,gAB,1,1); }  // hc += conv1_bias
    // time embedding: ts=silu(temb); tp = ts @ tprojWT + tpb
    { N1 n{(uint32_t)Temb}; op(1,{&btemb,&ts},&n,4,((uint32_t)Temb+255)/256,1,1); }
    { MM mm{1,(uint32_t)C,(uint32_t)Temb}; op(3,{&ts,&tpw,&tp},&mm,12,((uint32_t)C+127)/128,1,1); }
    { AB ab{(uint32_t)C,1}; op(4,{&tp,&tpb},&ab,8,((uint32_t)C+255)/256,1,1); }  // tp += tproj_bias
    { AB ab{(uint32_t)C,HW}; op(4,{&hc,&tp},&ab,8,gAB,1,1); }   // hc += tp (broadcast по каналам)
    op(0,{&hc,&n2w,&n2b,&h2},&gn,16,G,1,1);             // h2 = groupnorm(hc)
    { N1 n{N}; op(1,{&h2,&h2},&n,4,gSilu,1,1); }        // h2 = silu(h2)
    op(2,{&h2,&c2w,&h},&cv,28,cgx,cgy,cgz);             // h = conv2(h2)  (переиспользуем h)
    { AB ab{(uint32_t)C,HW}; op(4,{&h,&c2b},&ab,8,gAB,1,1); }   // h += conv2_bias
    { N1 n{N}; op(5,{&bx,&h,&out},&n,4,gSilu,1,1); }    // out = x + h

    // сверка
    std::vector<__fp16> hy(N); c.download(out,hy.data(),(VkDeviceSize)N*2);
    double se=0,sr=0; for (uint32_t i=0;i<N;i+=37){ se+=fabsf((float)yref[i]-(float)hy[i]); sr+=fabsf((float)yref[i]); }
    double e=se/(sr+1e-6);
    LOG("RESNET C%d %dx%d corr=%.4f %s",C,H,W,e,e<0.03?"OK":"FAIL");
    c.destroy(); return e;
}

// ====================== JNI: ATTENTION ======================
extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchAttention(
        JNIEnv* env, jobject, jbyteArray spirv, jint H, jint seqQ, jint seqK, jint d, jint iters) {
    VkCtx c; if(!c.init()) return -1;
    jsize l=env->GetArrayLength(spirv); std::vector<uint8_t> spv(l); env->GetByteArrayRegion(spirv,0,l,(jbyte*)spv.data());
    VkDeviceSize szQ=(VkDeviceSize)H*seqQ*d*2, szK=(VkDeviceSize)H*seqK*d*2, szO=szQ;
    Buf bQ=c.alloc(szQ,true), bK=c.alloc(szK,true), bV=c.alloc(szK,true), bO=c.alloc(szO,true);
    std::vector<__fp16> hQ((size_t)H*seqQ*d), hK((size_t)H*seqK*d), hV((size_t)H*seqK*d);
    for(size_t i=0;i<hQ.size();i++) hQ[i]=(__fp16)(((i*131u+7u)%23u)*0.05f-0.55f);
    for(size_t i=0;i<hK.size();i++) hK[i]=(__fp16)(((i*61u+13u)%19u)*0.05f-0.45f);
    for(size_t i=0;i<hV.size();i++) hV[i]=(__fp16)(((i*97u+5u)%17u)*0.05f-0.4f);
    c.upload(bQ,hQ.data(),szQ); c.upload(bK,hK.data(),szK); c.upload(bV,hV.data(),szK);

    Kernel k; k.create(c,spv.data(),spv.size(),4,20); VkDescriptorSet ds=k.makeSet(c,{&bQ,&bK,&bV,&bO});
    float scale=1.0f/sqrtf((float)d);
    struct { uint32_t sQ,sK,d,H; float sc; } pc{(uint32_t)seqQ,(uint32_t)seqK,(uint32_t)d,(uint32_t)H,scale};
    uint32_t gx=((uint32_t)seqQ+63)/64, gy=(uint32_t)H;

    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,&pc,gx,gy,1); c.endCmd(cmd); }
    // верификация: эталонный softmax-attention на CPU для нескольких (h,i,k)
    { std::vector<__fp16> hO((size_t)H*seqQ*d); c.download(bO,hO.data(),szO);
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

    VkDeviceSize szX=(VkDeviceSize)Cin*H*W*2, szW=(VkDeviceSize)Cout*Kcol*2;
    VkDeviceSize szCol=(VkDeviceSize)Kcol*Ncol*2, szY=(VkDeviceSize)Cout*Ncol*2;
    Buf bX=c.alloc(szX,true), bW=c.alloc(szW,true), bCol=c.alloc(szCol,true), bY=c.alloc(szY,true);
    std::vector<__fp16> hX((size_t)Cin*H*W), hW((size_t)Cout*Kcol);
    for (size_t i=0;i<hX.size();i++) hX[i]=(__fp16)(((i*131u+7u)%17u)*0.02f-0.16f);
    for (size_t i=0;i<hW.size();i++) hW[i]=(__fp16)(((i*61u+13u)%19u)*0.02f-0.18f);
    c.upload(bX,hX.data(),szX); c.upload(bW,hW.data(),szW);

    Kernel ic; ic.create(c,icSpv.data(),icSpv.size(),2,32);
    VkDescriptorSet icDs=ic.makeSet(c,{&bX,&bCol});
    uint32_t icPc[8]={(uint32_t)Cin,(uint32_t)H,(uint32_t)W,(uint32_t)KH,(uint32_t)KW,(uint32_t)pad,Ncol,Kcol};
    uint32_t icG=(Kcol*Ncol+255)/256;

    Kernel mm; mm.create(c,mmSpv.data(),mmSpv.size(),3,12);
    VkDescriptorSet mmDs=mm.makeSet(c,{&bW,&bCol,&bY});  // A=W[M,K], B=Col[K,N], C=Y[M,N]
    uint32_t mmPc[3]={Mc,Ncol,Kcol};
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
    { std::vector<__fp16> hY((size_t)Cout*Ncol); c.download(bY,hY.data(),szY);
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

    VkDeviceSize szX=(VkDeviceSize)Cin*H*W*2, szW=(VkDeviceSize)Cout*Cin*KH*KW*2, szY=(VkDeviceSize)Cout*H*W*2;
    Buf bX=c.alloc(szX,true), bW=c.alloc(szW,true), bY=c.alloc(szY,true);
    std::vector<__fp16> hX((size_t)Cin*H*W), hW((size_t)Cout*Cin*KH*KW);
    for (size_t i=0;i<hX.size();i++) hX[i]=(__fp16)(((i*131u+7u)%17u)*0.02f-0.16f);
    for (size_t i=0;i<hW.size();i++) hW[i]=(__fp16)(((i*61u+13u)%19u)*0.02f-0.18f);
    c.upload(bX,hX.data(),szX); c.upload(bW,hW.data(),szW);

    Kernel k; k.create(c,spv.data(),spvLen,3,28);  // push: Cin,Cout,H,W,KH,KW,pad
    VkDescriptorSet ds=k.makeSet(c,{&bX,&bW,&bY});
    uint32_t pc[7]={(uint32_t)Cin,(uint32_t)Cout,(uint32_t)H,(uint32_t)W,(uint32_t)KH,(uint32_t)KW,(uint32_t)pad};
    uint32_t gx=((uint32_t)W+15)/16, gy=((uint32_t)H+15)/16, gz=(uint32_t)Cout;

    { VkCommandBuffer cmd=c.beginCmd(); k.record(cmd,ds,pc,gx,gy,gz); c.endCmd(cmd); }
    { std::vector<__fp16> hY((size_t)Cout*H*W); c.download(bY,hY.data(),szY);
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
