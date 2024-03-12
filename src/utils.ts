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

// async function test_sort(sorter: GPUSorter, device: GPUDevice, queue: GPUQueue): boolean {
//     // simply runs a small sort and check if the sorting result is correct
//     let n = 8192; // means that 2 workgroups are needed for sorting
//     let scrambled_data: Vec<f32> = (0..n).rev().map(|x| x as f32).collect();
//     let sorted_data: Vec<f32> = (0..n).map(|x| x as f32).collect();

//     let sort_buffers = sorter.create_sort_buffers(device, NonZeroU32::new(n).unwrap());

//     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
//         label: Some("GPURSSorter test_sort"),
//     });
//     upload_to_buffer(
//         &mut encoder,
//         &sort_buffers.keys(),
//         device,
//         scrambled_data.as_slice(),
//     );

//     sorter.sort(&mut encoder, queue, &sort_buffers,None);
//     let idx = queue.submit([encoder.finish()]);
//     device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));

//     let sorted = download_buffer::<f32>(
//         &sort_buffers.keys(),
//         device,
//         queue,
//         0..sort_buffers.keys_valid_size(),
//     )
//     .await;
//     return sorted.into_iter().zip(sorted_data.into_iter()).all(|(a,b)|a==b);
// }

/// Function guesses the best subgroup size by testing the sorter with
/// subgroup sizes 1,8,16,32,64,128 and returning the largest subgroup size that worked.
// async function guess_workgroup_size(device: GPUDevice, queue: GPUQueue): number {
//     let cur_sorter: GPUSorter = null;

//     console.log("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");

//     let best = null;
//     [1, 8, 16, 32, 64, 128].every((subgroup_size) => {
//         console.log("Checking sorting with subgroupsize {}", subgroup_size);

//         cur_sorter = new GPUSorter(device, subgroup_size);
//         let sort_success = test_sort(cur_sorter, device, queue).await;

//         console.log("{} worked: {}", subgroup_size, sort_success);

//         if (!sort_success) {
//             return false;
//         } else {
//             best = subgroup_size;
//             return true;
//         }
//     });
//     return best;
// }