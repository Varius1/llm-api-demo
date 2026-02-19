import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

// --- Модели данных для OpenRouter API (совместим с OpenAI) ---

@Serializable
data class ChatMessage(
    val role: String,
    val content: String,
)

@Serializable
data class ChatRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val temperature: Double? = null,
)

@Serializable
data class ChatChoice(
    val message: ChatMessage,
)

@Serializable
data class ChatResponse(
    val choices: List<ChatChoice> = emptyList(),
    val error: ChatError? = null,
)

@Serializable
data class ChatError(
    val message: String = "Unknown error",
    val code: Int? = null,
)

// --- Конфигурация ---

private const val OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
private const val DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"

private val json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = false
}

private val client = OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .readTimeout(60, TimeUnit.SECONDS)
    .build()

// --- Основная логика ---

fun sendMessage(apiKey: String, messages: List<ChatMessage>, model: String, temperature: Double?): String {
    val requestBody = json.encodeToString(
        ChatRequest(model = model, messages = messages, temperature = temperature)
    )

    val request = Request.Builder()
        .url(OPENROUTER_URL)
        .addHeader("Authorization", "Bearer $apiKey")
        .addHeader("Content-Type", "application/json")
        .post(requestBody.toRequestBody("application/json".toMediaType()))
        .build()

    client.newCall(request).execute().use { response ->
        val body = response.body?.string() ?: error("Пустой ответ от сервера")

        if (!response.isSuccessful) {
            error("HTTP ${response.code}: $body")
        }

        val chatResponse = json.decodeFromString<ChatResponse>(body)

        if (chatResponse.error != null) {
            error("API ошибка: ${chatResponse.error.message}")
        }

        return chatResponse.choices.firstOrNull()?.message?.content
            ?: error("Нет ответа в choices")
    }
}

fun main() {
    val apiKey = System.getenv("OPENROUTER_API_KEY")
        ?: run {
            print("Введите ваш OpenRouter API ключ: ")
            readlnOrNull()?.trim() ?: error("API ключ не введён")
        }

    val model = System.getenv("LLM_MODEL") ?: DEFAULT_MODEL
    var temperature = System.getenv("LLM_TEMPERATURE")?.toDoubleOrNull()

    println("=== LLM CLI Chat ===")
    println("Модель: $model")
    println("Температура: ${temperature ?: "по умолчанию (1.0)"}")
    println("Команды: /temp 0.7 — температура, exit — выход")
    println("Введите сообщение (двойной Enter — отправить)")
    println()

    while (true) {
        print("Вы: ")
        val lines = mutableListOf<String>()
        var emptyCount = 0
        while (true) {
            val line = readlnOrNull() ?: break
            if (line.isEmpty()) {
                emptyCount++
                if (emptyCount >= 2) break
                lines.add(line)
            } else {
                emptyCount = 0
                val trimmed = line.trim()
                if (lines.isEmpty() && trimmed.startsWith("/temp")) {
                    val newTemp = trimmed.removePrefix("/temp").trim().toDoubleOrNull()
                    temperature = newTemp
                    println("Температура: ${temperature ?: "по умолчанию (1.0)"}\n")
                    emptyCount = 2
                    break
                }
                if (lines.isEmpty() && (trimmed.equals("exit", ignoreCase = true) || trimmed.equals("quit", ignoreCase = true))) {
                    println("До свидания!")
                    return
                }
                lines.add(line)
            }
        }

        val input = lines.joinToString("\n").trim()

        if (input.isEmpty()) continue

        val messages = listOf(ChatMessage(role = "user", content = input))

        try {
            println("[DEBUG] temperature = $temperature")
            val reply = sendMessage(apiKey, messages, model, temperature)
            println("\nLLM:\n$reply\n")
        } catch (e: Exception) {
            println("\nОшибка: ${e.message}\n")
        }
    }
}
