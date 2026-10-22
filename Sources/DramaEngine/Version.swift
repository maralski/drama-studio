/// The semantic version of the DramaEngine library.
///
/// A major version changes when a public symbol is removed or changes meaning,
/// a minor version when one is added, and a patch version for fixes that keep
/// the public surface as it is. `CHANGELOG.md` at the repository root lists
/// each release.
public enum DramaEngineVersion: Sendable {
    /// The version as `major.minor.patch`, for example `"1.0.0"`.
    public static let string = "\(major).\(minor).\(patch)"
    /// The major version.
    public static let major = 1
    /// The minor version.
    public static let minor = 0
    /// The patch version.
    public static let patch = 1
}
