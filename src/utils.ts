import { GPUSorter } from "./sorter";

// https://github.com/denoland/webgpu-examples/blob/main/utils.ts
export interface BufferInit {
    label?: string;
    usage: number;
    contents: ArrayBuffer;
}

// ?
export function createBufferInit(device: GPUDevice, descriptor: BufferInit): GPUBuffer {
    const contents = new Uint32Array(descriptor.contents);

    const alignMask = 4 - 1;
    const paddedSize = Math.max(
        (contents.byteLength + alignMask) & ~alignMask,
        4
    );

    const buffer = device.createBuffer({
        label: descriptor.label,
        usage: descriptor.usage,
        mappedAtCreation: true,
        size: paddedSize
    });
    const data = new Uint32Array(buffer.getMappedRange());
    data.set(contents);
    buffer.unmap();

    return buffer;
}

export function upload_to_buffer(encoder: GPUCommandEncoder, buffer: GPUBuffer, device: GPUDevice, values: ArrayBuffer) {
    let staging_buffer: GPUBuffer = createBufferInit(device, {
        label: "Staging buffer",
        contents: values,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    } as BufferInit);
    encoder.copyBufferToBuffer(staging_buffer, 0, buffer, 0, staging_buffer.size);
}

async function test_sort(sorter: GPUSorter, device: GPUDevice, queue: GPUQueue): Promise<boolean> {
    // simply runs a small sort and check if the sorting result is correct
    let n = 8192; // means that 2 workgroups are needed for sorting
    let scrambled_data: Uint32Array = new Uint32Array([...Array(n).keys()].reverse());
    let sorted_data: Uint32Array = new Uint32Array([...Array(n).keys()]);

    let sort_buffers = sorter.create_sort_buffers(device, n);

    let encoder: GPUCommandEncoder = device.createCommandEncoder({
        label: "GPURSSorter test_sort",
    });
    upload_to_buffer(
        encoder,
        sort_buffers.keys(),
        device,
        scrambled_data
    );
    sorter.sort(encoder, queue, sort_buffers);
    
    const resultBuffer = device.createBuffer({
        label: "result buffer",
        size: sort_buffers.keys_valid_size(),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    encoder.copyBufferToBuffer(sort_buffers.keys(), 0, resultBuffer, 0, resultBuffer.size);
    queue.submit([encoder.finish()]);
    
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(resultBuffer.getMappedRange(0, sort_buffers.keys_valid_size()));
    let same: boolean = JSON.stringify(result) == JSON.stringify(sorted_data);
    resultBuffer.unmap();

    return same;
}

// Function guesses the best subgroup size by testing the sorter with
// subgroup sizes 1,8,16,32,64,128 and returning the largest subgroup size that worked.
export async function guess_workgroup_size(device: GPUDevice, queue: GPUQueue): Promise<number> {
    let cur_sorter: GPUSorter = null;

    // console.log("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");

    let best: number = null;
    let elements: Array<number> = [1, 8, 16, 32, 64, 128];
    for (let i = 0; i < elements.length; ++i) {
        let subgroup_size = elements[i];
        // console.log("Checking sorting with subgroupsize {}", subgroup_size);

        cur_sorter = new GPUSorter(device, subgroup_size);
        let sort_success = await test_sort(cur_sorter, device, queue);

        // console.log("{} worked: {}", subgroup_size, sort_success);
        if (sort_success) {
            best = subgroup_size;
        }
    }
    return best;
}