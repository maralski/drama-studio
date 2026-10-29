// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Draws the Drama Studio app icon on Apple's 1024-point macOS icon grid and
// writes AppIcon.iconset, which iconutil turns into AppIcon.icns.
//   swift App/Icon/make_icon.swift App/Icon/AppIcon.iconset
import AppKit

let outDir = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "AppIcon.iconset")
try? FileManager.default.removeItem(at: outDir)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat((hex >> 16) & 0xff) / 255, green: CGFloat((hex >> 8) & 0xff) / 255, blue: CGFloat(hex & 0xff) / 255, alpha: alpha)
}

func draw(_ ctx: CGContext, size: CGFloat) {
    let s = size / 1024
    ctx.scaleBy(x: s, y: s)
    let tile = CGRect(x: 100, y: 100, width: 824, height: 824)
    let shape = CGPath(roundedRect: tile, cornerWidth: 185, cornerHeight: 185, transform: nil)

    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 28, color: color(0x000000, 0.30))
    ctx.addPath(shape); ctx.setFillColor(color(0x000000)); ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.addPath(shape); ctx.clip()
    let background = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [color(0x3B1F78), color(0x8A2F86), color(0xE65B6E)] as CFArray, locations: [0, 0.55, 1])!
    ctx.drawLinearGradient(background, start: CGPoint(x: 160, y: 924), end: CGPoint(x: 864, y: 100), options: [])
    let glow = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [color(0xFFFFFF, 0.22), color(0xFFFFFF, 0)] as CFArray, locations: [0, 1])!
    ctx.drawRadialGradient(glow, startCenter: CGPoint(x: 512, y: 760), startRadius: 0, endCenter: CGPoint(x: 512, y: 760), endRadius: 520, options: [])

    let heights: [CGFloat] = [150, 300, 460, 300, 520, 360, 200]
    let barWidth: CGFloat = 62, gap: CGFloat = 34
    let total = CGFloat(heights.count) * barWidth + CGFloat(heights.count - 1) * gap
    var x = 512 - total / 2
    ctx.setShadow(offset: CGSize(width: 0, height: -6), blur: 18, color: color(0x1A0B3A, 0.35))
    for h in heights {
        let bar = CGRect(x: x, y: 512 - h / 2, width: barWidth, height: h)
        ctx.addPath(CGPath(roundedRect: bar, cornerWidth: barWidth / 2, cornerHeight: barWidth / 2, transform: nil))
        x += barWidth + gap
    }
    ctx.setFillColor(color(0xFFFFFF)); ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.addPath(CGPath(roundedRect: tile.insetBy(dx: 1.5, dy: 1.5), cornerWidth: 184, cornerHeight: 184, transform: nil))
    ctx.setStrokeColor(color(0xFFFFFF, 0.18)); ctx.setLineWidth(3); ctx.strokePath()
    ctx.restoreGState()
}

func png(_ pixels: Int, _ name: String) throws {
    let ctx = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    draw(ctx, size: CGFloat(pixels))
    let rep = NSBitmapImageRep(cgImage: ctx.makeImage()!)
    try rep.representation(using: .png, properties: [:])!.write(to: outDir.appendingPathComponent(name))
}

for base in [16, 32, 128, 256, 512] {
    try png(base, "icon_\(base)x\(base).png")
    try png(base * 2, "icon_\(base)x\(base)@2x.png")
}
print("wrote \(outDir.path)")
