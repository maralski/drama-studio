// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import WordRecognitionProtocol

/// Revision of the acceptance policy used for automatic audio references.
public enum WordVerificationPolicy {
    public static var currentRevision: Int { WordRecognitionAssets.policyRevision }
    public static var currentDefinition: String { WordRecognitionAssets.policyDefinition }
}
