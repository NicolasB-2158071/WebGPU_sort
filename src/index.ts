import { GPUSorter, SortBuffers } from "./sorter";
import { upload_to_buffer, guess_workgroup_size } from "./utils";

async function initWebGPU(): Promise<GPUDevice> {
    if (!navigator.gpu) {
        alert("WebGPU not supported!");
        return;
    }

    const adapter: GPUAdapter = await navigator.gpu.requestAdapter();
    const device: GPUDevice = await adapter.requestDevice({
        requiredLimits: {
            maxComputeWorkgroupStorageSize: 17500 // SCATTER_WG_SIZE = 256, reducing didn't help..
        }   
    });
    if (!adapter || !device) {
        alert("WebGPU not supported!");
        return;
    }
    return device;
}

async function main(): Promise<void> {
    let device: GPUDevice = await initWebGPU();
    let sorter: GPUSorter = new GPUSorter(device, await guess_workgroup_size(device, device.queue));

    let n: number = 10;
    let sort_buffers: SortBuffers = sorter.create_sort_buffers(device, n);

    // let keys_scrambled: Float32Array = new Float32Array([...Array(n).keys()].reverse());
    let keys_scrambled: Float32Array = new Float32Array(Array.from({length: n}, () => Math.random()));
    let values_scrambled: Float32Array = new Float32Array(Array.from({length: n}, () => Math.floor(Math.random() * n)));

    let encoder: GPUCommandEncoder = device.createCommandEncoder();

    // upload_to_buffer(
    //     encoder,
    //     sort_buffers.keys(),
    //     device,
    //     keys_scrambled
    // );
    device.queue.writeBuffer(sort_buffers.keys(), 0, keys_scrambled, 0, n); // For floats
    upload_to_buffer(
        encoder,
        sort_buffers.values(),
        device,
        values_scrambled
    );

    sorter.sort(encoder, device.queue, sort_buffers);

    const resultBuffer = device.createBuffer({
        label: "result buffer",
        size: sort_buffers.values().size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    encoder.copyBufferToBuffer(sort_buffers.values(), 0, resultBuffer, 0, resultBuffer.size);
    device.queue.submit([encoder.finish()]);

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(resultBuffer.getMappedRange(0, sort_buffers.keys_valid_size()));
    
    console.log("keys", JSON.parse(JSON.stringify(keys_scrambled))); // https://developer.mozilla.org/en-US/docs/Web/API/console/log_static#logging_objects
    console.log("values", values_scrambled);
    console.log("keys sorted", keys_scrambled.sort());
    console.log("values sorted", result);
    resultBuffer.unmap();
}

main();