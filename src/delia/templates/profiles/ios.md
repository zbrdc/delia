# iOS/Swift Profile

Load this profile for: iOS apps, SwiftUI, UIKit, Apple platform development.

## Project Structure

```
App/
├── App.swift                 # @main entry point
├── Features/
│   └── Auth/
│       ├── Views/
│       │   └── LoginView.swift
│       ├── ViewModels/
│       │   └── LoginViewModel.swift
│       └── Models/
│           └── User.swift
├── Core/
│   ├── Network/
│   ├── Storage/
│   └── Extensions/
├── Shared/
│   ├── Components/
│   └── Styles/
├── Resources/
│   └── Assets.xcassets
└── Tests/
```

## SwiftUI Views

```swift
import SwiftUI

struct UserProfileView: View {
    @StateObject private var viewModel = UserProfileViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Name", text: $viewModel.name)
                    TextField("Email", text: $viewModel.email)
                        .textContentType(.emailAddress)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                }

                Section {
                    Button("Save") {
                        Task {
                            await viewModel.save()
                            dismiss()
                        }
                    }
                    .disabled(!viewModel.isValid)
                }
            }
            .navigationTitle("Profile")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .alert("Error", isPresented: $viewModel.showError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }
}
```

## MVVM Pattern

```swift
import Foundation
import Combine

@MainActor
final class UserProfileViewModel: ObservableObject {
    @Published var name = ""
    @Published var email = ""
    @Published var isLoading = false
    @Published var showError = false
    @Published var errorMessage = ""

    var isValid: Bool {
        !name.isEmpty && email.contains("@")
    }

    private let userService: UserServiceProtocol
    private var cancellables = Set<AnyCancellable>()

    init(userService: UserServiceProtocol = UserService()) {
        self.userService = userService
    }

    func save() async {
        isLoading = true
        defer { isLoading = false }

        do {
            try await userService.updateProfile(name: name, email: email)
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }
}
```

## Async/Await & Error Handling

```swift
// Define errors
enum NetworkError: LocalizedError {
    case invalidURL
    case noData
    case decodingFailed
    case serverError(Int)

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid URL"
        case .noData: return "No data received"
        case .decodingFailed: return "Failed to decode response"
        case .serverError(let code): return "Server error: \(code)"
        }
    }
}

// Async service
protocol UserServiceProtocol {
    func fetchUser(id: String) async throws -> User
    func updateProfile(name: String, email: String) async throws
}

final class UserService: UserServiceProtocol {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func fetchUser(id: String) async throws -> User {
        guard let url = URL(string: "https://api.example.com/users/\(id)") else {
            throw NetworkError.invalidURL
        }

        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.noData
        }

        guard 200..<300 ~= httpResponse.statusCode else {
            throw NetworkError.serverError(httpResponse.statusCode)
        }

        return try JSONDecoder().decode(User.self, from: data)
    }
}
```

## Data Models

```swift
import Foundation

struct User: Codable, Identifiable, Equatable {
    let id: String
    var name: String
    var email: String
    let createdAt: Date

    enum CodingKeys: String, CodingKey {
        case id, name, email
        case createdAt = "created_at"
    }
}

// Use extensions for computed properties
extension User {
    var displayName: String {
        name.isEmpty ? email : name
    }

    var initials: String {
        name.split(separator: " ")
            .prefix(2)
            .compactMap { $0.first }
            .map(String.init)
            .joined()
    }
}
```

## Core Data / SwiftData

```swift
import SwiftData

@Model
final class Task {
    var title: String
    var isCompleted: Bool
    var createdAt: Date

    init(title: String, isCompleted: Bool = false) {
        self.title = title
        self.isCompleted = isCompleted
        self.createdAt = Date()
    }
}

// Usage in View
struct TaskListView: View {
    @Query(sort: \Task.createdAt, order: .reverse)
    private var tasks: [Task]

    @Environment(\.modelContext) private var modelContext

    var body: some View {
        List(tasks) { task in
            TaskRow(task: task)
        }
    }

    func addTask(title: String) {
        let task = Task(title: title)
        modelContext.insert(task)
    }
}
```

## Keychain Security

```swift
import Security

final class KeychainService {
    static let shared = KeychainService()

    func save(_ data: Data, for key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    func load(for key: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess, let data = result as? Data else {
            throw KeychainError.loadFailed(status)
        }

        return data
    }
}
```

## Best Practices

```
ALWAYS:
- Use @MainActor for UI-related code
- Handle loading/error states in ViewModels
- Follow Human Interface Guidelines
- Support Dark Mode and Dynamic Type
- Use Keychain for sensitive data

AVOID:
- Force unwrapping (use guard/if let)
- Massive view controllers/views
- Blocking the main thread
- Hardcoded strings (use Localizable.strings)
```

