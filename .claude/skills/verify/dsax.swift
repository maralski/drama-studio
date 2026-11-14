// Drives one app by process id through the accessibility API, without taking focus.
//   dsax tree <pid>                 every element in the app's windows: role | title | description | help | value
//   dsax press <pid> <text>         AXPress the first element whose title, description, help or identifier is <text>
//   dsax set <pid> <role> <text>    set the value of the first element with that role, e.g. AXTextArea
//   dsax menu <pid> <title>...      AXPress a menu item by its path, e.g. Speech Engine Ming; "list" as the last word lists
//   dsax windows <pid>              the app's windows: id, size, on or off screen, title
import AppKit
import ApplicationServices

func fail(_ message: String) -> Never { FileHandle.standardError.write(Data((message + "\n").utf8)); exit(1) }
func attribute<T>(_ element: AXUIElement, _ name: String) -> T? {
    var value: CFTypeRef?
    return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value as? T : nil
}
func children(_ element: AXUIElement) -> [AXUIElement] { attribute(element, kAXChildrenAttribute) ?? [] }
func text(_ element: AXUIElement, _ name: String) -> String { (attribute(element, name) as String?) ?? "" }
func labels(_ element: AXUIElement) -> [String] {
    [kAXTitleAttribute, kAXDescriptionAttribute, kAXHelpAttribute, kAXIdentifierAttribute].map { text(element, $0) }
}
func walk(_ element: AXUIElement, depth: Int = 0, _ visit: (AXUIElement, Int) -> Bool) -> Bool {
    if visit(element, depth) { return true }
    for child in children(element) where walk(child, depth: depth + 1, visit) { return true }
    return false
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count >= 2, let pid = pid_t(arguments[1]) else { fail("usage: dsax tree|press|set|menu|windows <pid> ...") }
guard AXIsProcessTrusted() else { fail("This terminal has no Accessibility permission (System Settings > Privacy & Security > Accessibility).") }
let app = AXUIElementCreateApplication(pid)
let windows: [AXUIElement] = attribute(app, kAXWindowsAttribute) ?? []

switch arguments[0] {
case "tree":
    for window in windows {
        _ = walk(window) { element, depth in
            let value = (attribute(element, kAXValueAttribute) as AnyObject?).map { String(describing: $0).prefix(120) } ?? ""
            let line = [text(element, kAXRoleAttribute)] + labels(element).prefix(3) + [value.replacingOccurrences(of: "\n", with: "⏎")]
            print(String(repeating: "  ", count: depth) + line.joined(separator: " | "))
            return false
        }
    }
case "press":
    guard arguments.count == 3 else { fail("usage: dsax press <pid> <text>") }
    let found = windows.contains { window in
        walk(window) { element, _ in
            guard labels(element).contains(arguments[2]) else { return false }
            return AXUIElementPerformAction(element, kAXPressAction as CFString) == .success
        }
    }
    if !found { fail("no pressable element is labelled \(arguments[2])") }
case "set":
    guard arguments.count == 4 else { fail("usage: dsax set <pid> <role> <text>") }
    let found = windows.contains { window in
        walk(window) { element, _ in
            guard text(element, kAXRoleAttribute) == arguments[2] else { return false }
            return AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, arguments[3] as CFString) == .success
        }
    }
    if !found { fail("no settable \(arguments[2]) element") }
case "menu":
    guard arguments.count >= 3, var item: AXUIElement = attribute(app, kAXMenuBarAttribute) else { fail("usage: dsax menu <pid> <title>...") }
    func items(_ element: AXUIElement) -> [AXUIElement] {
        children(element).flatMap { text($0, kAXRoleAttribute) == "AXMenu" ? children($0) : [$0] }
    }
    for title in arguments.dropFirst(2) {
        if title == "list" {
            for child in items(item) {
                let mark = (attribute(child, kAXMenuItemMarkCharAttribute) as String?) ?? " "
                let enabled = (attribute(child, kAXEnabledAttribute) as Bool?) ?? true
                print("\(mark) \(text(child, kAXTitleAttribute))\(enabled ? "" : "  (disabled)")")
            }
            exit(0)
        }
        guard let next = items(item).first(where: { text($0, kAXTitleAttribute) == title }) else { fail("no menu item \(title)") }
        item = next
    }
    guard AXUIElementPerformAction(item, kAXPressAction as CFString) == .success else { fail("the menu item did not accept a press") }
case "windows":
    let list = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] ?? []
    for window in list where (window[kCGWindowOwnerPID as String] as? pid_t) == pid && (window[kCGWindowLayer as String] as? Int) == 0 {
        let bounds = window[kCGWindowBounds as String] as? [String: CGFloat] ?? [:]
        let on = (window[kCGWindowIsOnscreen as String] as? Bool) == true
        print("\(window[kCGWindowNumber as String] ?? 0) \(Int(bounds["Width"] ?? 0))x\(Int(bounds["Height"] ?? 0)) \(on ? "on" : "off") \(window[kCGWindowName as String] as? String ?? "")")
    }
default:
    fail("unknown command \(arguments[0])")
}
