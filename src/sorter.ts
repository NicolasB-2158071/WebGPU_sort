import radix_shader from "./radix_sort.wgsl?raw";
import { createBufferInit, BufferInit } from "./utils";

// IMPORTANT: the following constants have to be synced with the numbers in radix_sort.wgsl

/// workgroup size of histogram shader
const HISTOGRAM_WG_SIZE: number = 256;

/// one thread operates on 2 prefixes at the same time
const PREFIX_WG_SIZE: number = 1 << 7;

/// scatter compute shader work group size
const SCATTER_WG_SIZE: number = 1 << 8;

/// we sort 8 bits per pass
const RS_RADIX_LOG2: number = 8;

/// 256 entries into the radix table
const RS_RADIX_SIZE: number = 1 << RS_RADIX_LOG2;

/// number of bytes our keys and values have
const RS_KEYVAL_SIZE: number = 32 / RS_RADIX_LOG2;

/// TODO describe me
const RS_HISTOGRAM_BLOCK_ROWS: number = 15;

/// DO NOT CHANGE, shader assume this!!!
const RS_SCATTER_BLOCK_ROWS: number = RS_HISTOGRAM_BLOCK_ROWS;

/// number of elements scattered by one work group
const SCATTER_BLOCK_KVS: number = HISTOGRAM_WG_SIZE * RS_SCATTER_BLOCK_ROWS;

/// number of elements scattered by one work group
const HISTO_BLOCK_KVS: number = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;

/// bytes per value
/// currently only 4 byte values are allowed
const BYTES_PER_PAYLOAD_ELEM: number = 4;

/// number of passed used for sorting
/// we sort 8 bits per pass so 4 passes are required for a 32 bit value
const NUM_PASSES: number = BYTES_PER_PAYLOAD_ELEM;

function scatter_blocks_ru(n: number): number {
    return Math.floor((n + SCATTER_BLOCK_KVS - 1) / SCATTER_BLOCK_KVS);
}

/// number of histogram blocks required
function histo_blocks_ru(n: number): number {
    return Math.floor((scatter_blocks_ru(n) * SCATTER_BLOCK_KVS + HISTO_BLOCK_KVS - 1) / HISTO_BLOCK_KVS);
}


/// keys buffer must be multiple of HISTO_BLOCK_KVS
function keys_buffer_size(n: number): number {
    return histo_blocks_ru(n) * HISTO_BLOCK_KVS;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------

interface SorterState {
    /// number of first n keys that will be sorted
    num_keys: number,
    padded_size: number,
    even_pass: number,
    odd_pass: number
}

/// Struct containing all buffers necessary for sorting.
/// The key and value buffers can be read and written.
export class SortBuffers {
    /// keys that are sorted
    public keys_a: GPUBuffer
    /// intermediate key buffer for sorting
    public keys_b: GPUBuffer
    /// value/payload buffer that is sorted
    public payload_a: GPUBuffer
    /// intermediate value buffer for sorting
    public payload_b: GPUBuffer

    /// buffer used to store intermediate results like histograms and scatter partitions
    public internal_mem_buffer: GPUBuffer

    /// state buffer used for sorting
    public state_bufferVar: GPUBuffer  // rename because of ts

    /// bind group used for sorting
    public bind_group: GPUBindGroup

    // number of key-value pairs
    public length: number

    constructor(
        keys_a: GPUBuffer, keys_b: GPUBuffer,
        payload_a: GPUBuffer, payload_b: GPUBuffer,
        internal_mem_buffer: GPUBuffer, state_buffer: GPUBuffer,
        bind_group: GPUBindGroup, length: number) {
        this.keys_a = keys_a; this.keys_b = keys_b;
        this.payload_a = payload_a; this.payload_b = payload_b;
        this.internal_mem_buffer = internal_mem_buffer; this.state_bufferVar = state_buffer;
        this.bind_group = bind_group; this.length = length;
    }

    /// number of key-value pairs that can be stored in this buffer
    public len(): number {
        return this.length;
    }

    /// Buffer storing the keys values.
    /// 
    /// **WARNING**: this buffer has padding bytes at the end
    ///        use [SortBuffers::keys_valid_size] to get the valid size.
    public keys(): GPUBuffer {
        return this.keys_a;
    }

    /// The keys buffer has padding bytes.
    /// This function returns the number of bytes without padding
    public keys_valid_size(): number {
        return this.len() * RS_KEYVAL_SIZE;
    }

    /// Buffer containing the values
    public values(): GPUBuffer {
        return this.payload_a;
    }

    /// Buffer containing a [SorterState]
    public state_buffer(): GPUBuffer {
        return this.state_bufferVar;
    }
}

interface KeyvalBuffers {
    keys: GPUBuffer,
    keys_aux: GPUBuffer,
    payload: GPUBuffer,
    payload_aux: GPUBuffer
}

// ---------------------------------------------------------------------------------------------------------------------------------------------


export class GPUSorter {
    private zero_p: GPUComputePipeline
    private histogram_p: GPUComputePipeline
    private prefix_p: GPUComputePipeline
    private scatter_even_p: GPUComputePipeline
    private scatter_odd_p: GPUComputePipeline

    constructor(device: GPUDevice, subgroup_size: number) {
        // special variables for scatter shade
        let histogram_sg_size: number = subgroup_size;
        let rs_sweep_0_size: number = Math.floor(RS_RADIX_SIZE / histogram_sg_size);
        let rs_sweep_1_size: number = Math.floor(rs_sweep_0_size / histogram_sg_size);
        let rs_sweep_2_size: number = Math.floor(rs_sweep_1_size / histogram_sg_size);
        let rs_sweep_size: number = rs_sweep_0_size + rs_sweep_1_size + rs_sweep_2_size;
        let _rs_smem_phase_1: number = RS_RADIX_SIZE + RS_RADIX_SIZE + rs_sweep_size;
        let rs_smem_phase_2: number = RS_RADIX_SIZE + RS_SCATTER_BLOCK_ROWS * SCATTER_WG_SIZE;
        // rs_smem_phase_2 will always be larger, so always use phase2
        let rs_mem_dwords: number = rs_smem_phase_2;
        let rs_mem_sweep_0_offset: number = 0;
        let rs_mem_sweep_1_offset: number = rs_mem_sweep_0_offset + rs_sweep_0_size;
        let rs_mem_sweep_2_offset: number = rs_mem_sweep_1_offset + rs_sweep_1_size;

        let bind_group_layout: GPUBindGroupLayout = this.bind_group_layout(device);
        let pipeline_layout: GPUPipelineLayout = device.createPipelineLayout({
            label: "radix sort pipeline layout",
            bindGroupLayouts: [bind_group_layout]
        } as GPUPipelineLayoutDescriptor);

        // TODO replace with this with pipeline-overridable constants once they are available
        let shader_w_const = `
        const histogram_sg_size: u32 = ${histogram_sg_size}u;\n\
        const histogram_wg_size: u32 = ${HISTOGRAM_WG_SIZE}u;\n\
        const rs_radix_log2: u32 = ${RS_RADIX_LOG2}u;\n\
        const rs_radix_size: u32 = ${RS_RADIX_SIZE}u;\n\
        const rs_keyval_size: u32 = ${RS_KEYVAL_SIZE}u;\n\
        const rs_histogram_block_rows: u32 = ${RS_HISTOGRAM_BLOCK_ROWS}u;\n\
        const rs_scatter_block_rows: u32 = ${RS_SCATTER_BLOCK_ROWS}u;\n\
        const rs_mem_dwords: u32 = ${rs_mem_dwords}u;\n\
        const rs_mem_sweep_0_offset: u32 = ${rs_mem_sweep_0_offset}u;\n\
        const rs_mem_sweep_1_offset: u32 = ${rs_mem_sweep_1_offset}u;\n\
        const rs_mem_sweep_2_offset: u32 = ${rs_mem_sweep_2_offset}u;\n${radix_shader}`;

        let shader_code: string = shader_w_const
        .replaceAll(
            "{histogram_wg_size}",
            HISTOGRAM_WG_SIZE.toString(),
        )
        .replaceAll("{prefix_wg_size}", PREFIX_WG_SIZE.toString())
        .replaceAll("{scatter_wg_size}", SCATTER_WG_SIZE.toString());

        let shader: GPUShaderModule = device.createShaderModule( {
            label: "Radix sort shader",
            code: shader_code
        } as GPUShaderModuleDescriptor);
         this.zero_p = device.createComputePipeline({
            label: "Zero the histograms",
            layout: pipeline_layout,
            compute: {
                module: shader,
                entryPoint: "zero_histograms"
            }
        } as GPUComputePipelineDescriptor);
        this.histogram_p = device.createComputePipeline({
            label: "calculate_histogram",
            layout: pipeline_layout,
            compute: {
                module: shader,
                entryPoint: "calculate_histogram"
            }
        } as GPUComputePipelineDescriptor);
        this.prefix_p = device.createComputePipeline({
            label: "prefix_histogram",
            layout: pipeline_layout,
            compute: {
                module: shader,
                entryPoint: "prefix_histogram"
            }
        } as GPUComputePipelineDescriptor);
        this.scatter_even_p = device.createComputePipeline({
            label: "scatter_even",
            layout: pipeline_layout,
            compute: {
                module: shader,
                entryPoint: "scatter_even"
            }
        } as GPUComputePipelineDescriptor);
        this.scatter_odd_p = device.createComputePipeline({
            label: "scatter_odd",
            layout: pipeline_layout,
            compute: {
                module: shader,
                entryPoint: "scatter_odd"
            }
        } as GPUComputePipelineDescriptor);
        // console.log(shader_code);
    }

    private bind_group_layout(device: GPUDevice): GPUBindGroupLayout {
        return device.createBindGroupLayout({
            label: "radix sort bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: false,
                        minBindingSize: 16 // Size of SorterState, ?
                    } as GPUBufferBindingLayout
                } as GPUBindGroupLayoutEntry,
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: false
                    } as GPUBufferBindingLayout
                } as GPUBindGroupLayoutEntry,
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: false
                    } as GPUBufferBindingLayout
                } as GPUBindGroupLayoutEntry,
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: false,
                    } as GPUBufferBindingLayout
                } as GPUBindGroupLayoutEntry,
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: false,
                    } as GPUBufferBindingLayout
                } as GPUBindGroupLayoutEntry,
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                        hasDynamicOffset: false,
                    } as GPUBufferBindingLayout
                } as GPUBindGroupLayoutEntry
            ]
        } as GPUBindGroupLayoutDescriptor);
    }

    private create_keyval_buffers(device: GPUDevice, length: number): KeyvalBuffers {
        // add padding so that our buffer size is a multiple of keys_per_workgroup
        let count_ru_histo: number = keys_buffer_size(length) * RS_KEYVAL_SIZE;

        // creating the two needed buffers for sorting
        let keys: GPUBuffer = device.createBuffer({
            label: "radix sort keys buffer",
            size: count_ru_histo * BYTES_PER_PAYLOAD_ELEM,
            usage: GPUBufferUsage.STORAGE
                | GPUBufferUsage.COPY_DST
                | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false
        } as GPUBufferDescriptor);

        // auxiliary buffer for keys
        let keys_aux: GPUBuffer = device.createBuffer({
            label: "radix sort keys auxiliary buffer",
            size: count_ru_histo * BYTES_PER_PAYLOAD_ELEM,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: false
        } as GPUBufferDescriptor);

        let payload_size: number = length * BYTES_PER_PAYLOAD_ELEM; // make sure that we have at least 1 byte of data;
        let payload: GPUBuffer = device.createBuffer({
            label: "radix sort payload buffer",
            size: payload_size,
            usage: GPUBufferUsage.STORAGE
                | GPUBufferUsage.COPY_DST
                | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false
        } as GPUBufferDescriptor);

         // auxiliary buffer for payload/values
        let payload_aux: GPUBuffer = device.createBuffer({
            label: "radix sort payload auxiliary buffer",
            size: payload_size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: false
        } as GPUBufferDescriptor);

        return {keys, keys_aux, payload, payload_aux};
    }

    // calculates and allocates a buffer that is sufficient for holding all needed information for
    // sorting. This includes the histograms and the temporary scatter buffer
    // @return: tuple containing [internal memory buffer (should be bound at shader binding 1, count_ru_histo (padded size needed for the keyval buffer)]
    private create_internal_mem_buffer(device: GPUDevice, length: number): GPUBuffer {
        // currently only a few different key bits are supported, maybe has to be extended

        // The "internal" memory map looks like this:
        //   +---------------------------------+ <-- 0
        //   | histograms[keyval_size]         |
        //   +---------------------------------+ <-- keyval_size                           * histo_size
        //   | partitions[scatter_blocks_ru-1] |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
        //   | workgroup_ids[keyval_size]      |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

        let scatter_blocks_ruVar: number = scatter_blocks_ru(length); // rename because of ts

        let histo_size: number = RS_RADIX_SIZE * 4; // size of u32, ?

        let internal_size: number = (RS_KEYVAL_SIZE + scatter_blocks_ruVar) * histo_size; // +1 safety

        let buffer = device.createBuffer({
            label: "Internal radix sort buffer",
            size: internal_size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: false
        } as GPUBufferDescriptor);
        return buffer;
    }

    private general_info_data(length: number): SorterState {
        return {
            num_keys: length,
            padded_size: keys_buffer_size(length),
            even_pass: 0,
            odd_pass: 0
        }
    }

    private record_calculate_histogram(bind_group: GPUBindGroup, length: number, encoder: GPUCommandEncoder): void {
        // as we only deal with 32 bit float values always 4 passes are conducted
        let hist_blocks_ru: number = histo_blocks_ru(length);

        {
            let pass: GPUComputePassEncoder = encoder.beginComputePass({
                label: "zeroing the histogram"
            } as GPUComputePassDescriptor);

            pass.setPipeline(this.zero_p);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroups(hist_blocks_ru, 1, 1);
            pass.end();
        }

        {
            let pass: GPUComputePassEncoder = encoder.beginComputePass({
                label: "calculate histogram"
            } as GPUComputePassDescriptor);

            pass.setPipeline(this.histogram_p);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroups(hist_blocks_ru, 1, 1);
            pass.end();
        }
 
    }

    private record_calculate_histogram_indirect(bind_group: GPUBindGroup, dispatch_buffer: GPUBuffer, encoder: GPUCommandEncoder): void {
        {
            let pass: GPUComputePassEncoder = encoder.beginComputePass({
                label: "zeroing the histogram"
            } as GPUComputePassDescriptor);

            pass.setPipeline(this.zero_p);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroupsIndirect(dispatch_buffer, 0);
        }

        {
            let pass: GPUComputePassEncoder = encoder.beginComputePass({
                label: "calculate histogram"
            } as GPUComputePassDescriptor);

            pass.setPipeline(this.histogram_p);
            pass.setBindGroup(0, bind_group);
            pass.dispatchWorkgroupsIndirect(dispatch_buffer, 0);
        }
    }

    // There does not exist an indirect histogram dispatch as the number of prefixes is determined by the amount of passes
    private record_prefix_histogram(bind_group: GPUBindGroup, encoder: GPUCommandEncoder): void {
        let pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "prefix histogram",
        } as GPUComputePassDescriptor);

        pass.setPipeline(this.prefix_p);
        pass.setBindGroup(0, bind_group);
        pass.dispatchWorkgroups(NUM_PASSES, 1, 1);
        pass.end();
    }

    private record_scatter_keys(bind_group: GPUBindGroup, length: number, encoder: GPUCommandEncoder): void {
        let scatter_blocks_ruVar: number = scatter_blocks_ru(length); // rename because of ts

        let pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "Scatter keyvals"
        } as GPUComputePassDescriptor);

        pass.setBindGroup(0, bind_group);
        pass.setPipeline(this.scatter_even_p);
        pass.dispatchWorkgroups(scatter_blocks_ruVar, 1, 1);

        pass.setPipeline(this.scatter_odd_p);
        pass.dispatchWorkgroups(scatter_blocks_ruVar, 1, 1);

        pass.setPipeline(this.scatter_even_p);
        pass.dispatchWorkgroups(scatter_blocks_ruVar, 1, 1);

        pass.setPipeline(this.scatter_odd_p);
        pass.dispatchWorkgroups(scatter_blocks_ruVar, 1, 1);
        pass.end();
    }

    private record_scatter_keys_indirect(bind_group: GPUBindGroup, dispatch_buffer: GPUBuffer, encoder: GPUCommandEncoder): void {
        let pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "radix sort scatter keyvals"
        } as GPUComputePassDescriptor);

        pass.setBindGroup(0, bind_group);
        pass.setPipeline(this.scatter_even_p);
        pass.dispatchWorkgroupsIndirect(dispatch_buffer, 0);

        pass.setPipeline(this.scatter_odd_p);
        pass.dispatchWorkgroupsIndirect(dispatch_buffer, 0);

        pass.setPipeline(this.scatter_even_p);
        pass.dispatchWorkgroupsIndirect(dispatch_buffer, 0);

        pass.setPipeline(this.scatter_odd_p);
        pass.dispatchWorkgroupsIndirect(dispatch_buffer, 0);
    }

    /// Writes sort commands to command encoder.
    /// If sort_first_n is not none one the first n elements are sorted
    /// otherwise everything is sorted.
    ///
    /// **IMPORTANT**: if less than the whole buffer is sorted the rest of the keys buffer will be be corrupted
    public sort(encoder: GPUCommandEncoder, queue: GPUQueue, sort_buffers: SortBuffers, sort_first_n: number = null): void {
        let bind_group: GPUBindGroup = sort_buffers.bind_group;
        let num_elements: number = (sort_first_n != null) ? sort_first_n : sort_buffers.len();

        // write number of elements to buffer
        // queue.writeBuffer(sort_buffers.state_bufferVar, 0, new Uint32Array([num_elements])); // ?

        this.record_calculate_histogram(bind_group, num_elements, encoder);
        this.record_prefix_histogram(bind_group, encoder);
        this.record_scatter_keys(bind_group, num_elements, encoder);
    }

    /// Initiates sorting with an indirect call.
    /// The dispatch buffer must contain the struct [wgpu::util::DispatchIndirectArgs].
    ///
    /// number of y and z workgroups must be 1 
    ///
    /// x = (N + [HISTO_BLOCK_KVS]- 1 )/[HISTO_BLOCK_KVS], 
    /// where N are the first N elements to be sorted
    ///
    /// [SortBuffers::state_buffer] contains the number of keys that will be sorted.
    /// This is set to sort the whole buffer by default.
    ///
    /// **IMPORTANT**: if less than the whole buffer is sorted the rest of the keys buffer will most likely be corrupted. 
    public sort_indirect(encoder: GPUCommandEncoder, sort_buffers: SortBuffers, dispatch_buffer: GPUBuffer): void {
        let bind_group: GPUBindGroup = sort_buffers.bind_group;

        this.record_calculate_histogram_indirect(bind_group, dispatch_buffer, encoder);
        this.record_prefix_histogram(bind_group, encoder);
        this.record_scatter_keys_indirect(bind_group, dispatch_buffer, encoder);
    }

    /// creates all buffers necessary for sorting
    public create_sort_buffers(device: GPUDevice, length: number): SortBuffers {
        let keyValBuffer: KeyvalBuffers = this.create_keyval_buffers(device, length);
        let internal_mem_buffer: GPUBuffer = this.create_internal_mem_buffer(device, length);

        let uniform_infos: SorterState = this.general_info_data(length);
        let uniform_buffer: GPUBuffer = createBufferInit(device, { // ?
            label: "radix sort uniform buffer",
            contents: new Uint32Array([uniform_infos.num_keys, uniform_infos.padded_size, uniform_infos.even_pass, uniform_infos.odd_pass]),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        } as BufferInit);
        let bind_group = device.createBindGroup({
            label: "radix sort bind group",
            layout: this.bind_group_layout(device),
            entries: [
                {
                    binding: 0,
                    resource: {buffer: uniform_buffer} // ?
                } as GPUBindGroupEntry,
                {
                    binding: 1,
                    resource: {buffer: internal_mem_buffer}
                } as GPUBindGroupEntry,
                {
                    binding: 2,
                    resource: {buffer: keyValBuffer.keys}
                } as GPUBindGroupEntry,
                {
                    binding: 3,
                    resource: {buffer: keyValBuffer.keys_aux}
                } as GPUBindGroupEntry,
                {
                    binding: 4,
                    resource: {buffer: keyValBuffer.payload}
                } as GPUBindGroupEntry,
                {
                    binding: 5,
                    resource: {buffer: keyValBuffer.payload_aux}
                } as GPUBindGroupEntry
            ]
        } as GPUBindGroupDescriptor);

        // return (uniform_buffer, bind_group);
        return new SortBuffers(
            keyValBuffer.keys,
            keyValBuffer.keys_aux,
            keyValBuffer.payload,
            keyValBuffer.payload_aux,
            internal_mem_buffer=internal_mem_buffer,
            uniform_buffer,
            bind_group,
            length
        );
    }
}