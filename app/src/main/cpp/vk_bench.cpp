// Минимальный Vulkan compute бенчмарк matmul — фундамент собственного GPU-backend.
// Цель: замерить реальные GFLOPS на Tensor G5 (PowerVR), доказать что свои шейдеры быстры.
#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "VkBench", __VA_ARGS__)
#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { LOG("VK error %d at %d", r, __LINE__); return -1.0; } } while(0)

static uint32_t findMemType(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

struct Buf { VkBuffer buf; VkDeviceMemory mem; };

// DEVICE_LOCAL буфер (быстрая GPU-память) — главный рычаг производительности compute.
static Buf makeBuf(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize sz, bool deviceLocal) {
    Buf b{};
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sz;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(dev, &bi, nullptr, &b.buf);
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, b.buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    VkMemoryPropertyFlags want = deviceLocal ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ai.memoryTypeIndex = findMemType(phys, req.memoryTypeBits, want);
    vkAllocateMemory(dev, &ai, nullptr, &b.mem);
    vkBindBufferMemory(dev, b.buf, b.mem, 0);
    return b;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_example_generet_1image_1ai_sd_VulkanBench_benchMatmul(
        JNIEnv* env, jobject, jbyteArray spirv, jint M, jint N, jint K, jint iters) {
    // 1. Instance
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    // 2. Physical device (первый = GPU)
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst, &nd, devs.data());
    VkPhysicalDevice phys = devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    LOG("GPU: %s", props.deviceName);

    // 3. Compute queue family
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }

    // проверка поддержки fp16 (shaderFloat16 + 16bit storage)
    VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures s16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    f16.pNext = &s16;
    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; feat2.pNext = &f16;
    vkGetPhysicalDeviceFeatures2(phys, &feat2);
    LOG("shaderFloat16=%d storage16=%d", f16.shaderFloat16, s16.storageBuffer16BitAccess);

    // 4. Logical device + queue (с включением fp16 features)
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.pNext = &feat2;  // включаем shaderFloat16 + 16bit storage
    VkDevice dev; VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    // 5. Buffers A,B,C — DEVICE_LOCAL, fp16 (2 байта/элемент)
    VkDeviceSize szA = (VkDeviceSize)M*K*2, szB = (VkDeviceSize)K*N*2, szC = (VkDeviceSize)M*N*2;
    Buf bA = makeBuf(phys, dev, szA, true), bB = makeBuf(phys, dev, szB, true), bC = makeBuf(phys, dev, szC, true);
    VkCommandPoolCreateInfo scpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; scpci.queueFamilyIndex=qfi;
    VkCommandPool stPool; vkCreateCommandPool(dev, &scpci, nullptr, &stPool);

    // host-копии A,B (fp16) для CPU-эталона корректности
    std::vector<__fp16> hA((size_t)M*K), hB((size_t)K*N);
    for (size_t i=0;i<hA.size();i++) hA[i]=(__fp16)(((i*131u+7u)%17u)*0.01f - 0.08f);
    for (size_t i=0;i<hB.size();i++) hB[i]=(__fp16)(((i*61u+13u)%19u)*0.01f - 0.09f);

    auto uploadData=[&](Buf& dst, VkDeviceSize sz, const void* src){
        Buf stg = makeBuf(phys, dev, sz, false);
        void* p; vkMapMemory(dev, stg.mem, 0, sz, 0, &p);
        memcpy(p, src, sz); vkUnmapMemory(dev, stg.mem);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=stPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VkCommandBuffer c; vkAllocateCommandBuffers(dev,&ai,&c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
        VkBufferCopy cp{0,0,sz}; vkCmdCopyBuffer(c, stg.buf, dst.buf, 1, &cp); vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c;
        vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(dev,stPool,1,&c); vkDestroyBuffer(dev,stg.buf,nullptr); vkFreeMemory(dev,stg.mem,nullptr);
    };
    uploadData(bA, szA, hA.data());
    uploadData(bB, szB, hB.data());

    auto upload=[&](Buf& dst, VkDeviceSize sz, float val, size_t n){
        Buf stg = makeBuf(phys, dev, sz, false);
        void* p; vkMapMemory(dev, stg.mem, 0, sz, 0, &p);
        for (size_t i=0;i<n;i++) ((__fp16*)p)[i]=(__fp16)val; vkUnmapMemory(dev, stg.mem);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=stPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VkCommandBuffer c; vkAllocateCommandBuffers(dev,&ai,&c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
        VkBufferCopy cp{0,0,sz}; vkCmdCopyBuffer(c, stg.buf, dst.buf, 1, &cp); vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c;
        vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(dev,stPool,1,&c); vkDestroyBuffer(dev,stg.buf,nullptr); vkFreeMemory(dev,stg.mem,nullptr);
    };
    (void)upload;  // A,B уже залиты через uploadData

    // 6. Shader module из SPIR-V
    jsize spvLen = env->GetArrayLength(spirv);
    std::vector<uint8_t> spvData(spvLen);
    env->GetByteArrayRegion(spirv, 0, spvLen, (jbyte*)spvData.data());
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spvLen; smci.pCode = (const uint32_t*)spvData.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    // 7. Descriptor set layout (3 storage buffers)
    VkDescriptorSetLayoutBinding bind[3]{};
    for (int i=0;i<3;i++){ bind[i].binding=i; bind[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bind[i].descriptorCount=1; bind[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount=3; dslci.pBindings=bind;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    // 8. Pipeline layout + push constants (M,N,K)
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

    // 9. Compute pipeline
    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage=VK_SHADER_STAGE_COMPUTE_BIT; ssci.module=sm; ssci.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage=ssci; cpci.layout=pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    // 10. Descriptor pool + set
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&psz;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsai, &ds);
    VkDescriptorBufferInfo dbi[3]={{bA.buf,0,szA},{bB.buf,0,szB},{bC.buf,0,szC}};
    VkWriteDescriptorSet wr[3]{};
    for(int i=0;i<3;i++){ wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet=ds; wr[i].dstBinding=i; wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo=&dbi[i]; }
    vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);

    // 11. Command pool + buffer
    VkCommandPoolCreateInfo cmpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmpci.queueFamilyIndex=qfi;
    VkCommandPool cmdPool; vkCreateCommandPool(dev, &cmpci, nullptr, &cmdPool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=cmdPool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);

    uint32_t pc[3]={(uint32_t)M,(uint32_t)N,(uint32_t)K};
    uint32_t gx=((uint32_t)N+127)/128, gy=((uint32_t)M+127)/128;  // тайл 128×128

    auto record=[&](){
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
        vkCmdDispatch(cmd, gx, gy, 1);
        vkEndCommandBuffer(cmd);
    };
    auto submit=[&](){
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
    };

    record(); submit();  // прогрев (компиляция шейдера драйвером)

    // === ВЕРИФИКАЦИЯ КОРРЕКТНОСТИ: скачать C, сверить с CPU-эталоном ===
    {
        Buf rb = makeBuf(phys, dev, szC, false);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=stPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VkCommandBuffer c; vkAllocateCommandBuffers(dev,&ai,&c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
        VkBufferCopy cp{0,0,szC}; vkCmdCopyBuffer(c, bC.buf, rb.buf, 1, &cp); vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c;
        vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(queue);
        void* p; vkMapMemory(dev, rb.mem, 0, szC, 0, &p);
        __fp16* C = (__fp16*)p;
        double sumErr=0, sumRef=0; int checked=0;
        // сверяем 16 элементов; ошибка нормируется на типичный масштаб (не на близкий к 0 ref)
        for (int s=0;s<16;s++){
            int r=(s*97+3)%M, col=(s*53+11)%N;
            float ref=0; for (int k=0;k<K;k++) ref += (float)hA[(size_t)r*K+k]*(float)hB[(size_t)k*N+col];
            float got=(float)C[(size_t)r*N+col];
            sumErr += fabsf(ref-got); sumRef += fabsf(ref); checked++;
        }
        double maxRel = sumErr/(sumRef+1e-6);  // относительная L1-ошибка по выборке
        vkUnmapMemory(dev, rb.mem);
        vkFreeCommandBuffers(dev,stPool,1,&c); vkDestroyBuffer(dev,rb.buf,nullptr); vkFreeMemory(dev,rb.mem,nullptr);
        LOG("CORRECTNESS %dx%dx%d: maxRelErr=%.4f (%d точек) %s", M,N,K, maxRel, checked, maxRel<0.02?"OK":"FAIL");
    }

    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<iters;i++){ vkResetCommandBuffer(cmd,0); record(); submit(); }
    auto t1=std::chrono::high_resolution_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count()/iters;
    double flops=2.0*(double)M*N*K;
    double gflops=flops/sec/1e9;
    LOG("matmul %dx%dx%d: %.3f ms, %.1f GFLOPS", M,N,K, sec*1000.0, gflops);

    vkDeviceWaitIdle(dev);
    vkDestroyInstance(inst, nullptr);  // упрощённая очистка (процесс бенча)
    return gflops;
}
