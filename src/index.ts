import { GPUSorter, SortBuffers } from "./sorter";
import { upload_to_buffer } from "./utils";

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
    let sorter: GPUSorter = new GPUSorter(device, 32);

    let n: number = 10;
    let sort_buffers: SortBuffers = sorter.create_sort_buffers(device, n);

    let keys_scrambled: Float32Array = new Float32Array([...Array(n).keys()].reverse());
    let values_scrambled: Float32Array = new Float32Array(Array.from({length: n}, () => Math.floor(Math.random() * n)));

    let encoder: GPUCommandEncoder = device.createCommandEncoder();

    upload_to_buffer(
        encoder,
        sort_buffers.keys(),
        device,
        keys_scrambled
    );
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

    console.log("values", values_scrambled);
    console.log("sorted values (reverse keys)", result);
    resultBuffer.unmap();
}

main();