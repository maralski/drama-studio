import Darwin
import Foundation

/// Memory counters for the load report. Metal wires the buffers a command
/// buffer binds, which the process footprint does not count; the host's
/// wired count does.
package enum SystemMemory {
    package static func wiredBytes() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(stats.wire_count) * UInt64(vm_kernel_page_size)
    }

    private static func taskVMInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? info : nil
    }

    package static func residentBytes() -> UInt64 { taskVMInfo().map { UInt64($0.resident_size) } ?? 0 }
    package static func footprintBytes() -> UInt64 { taskVMInfo().map { UInt64($0.phys_footprint) } ?? 0 }
}
