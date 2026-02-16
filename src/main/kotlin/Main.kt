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
private const val DEFAULT_MODEL = "openai/gpt-3.5-turbo"

private val json = Json { ignoreUnknownKeys = true }

private val client = OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .readTimeout(60, TimeUnit.SECONDS)
    .build()

// --- Основная логика ---

fun sendMessage(apiKey: String, messages: List<ChatMessage>, model: String): String {
    val requestBody = json.encodeToString(
        ChatRequest(model = model, messages = messages)
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

    println("=== LLM CLI Chat ===")
    println("Модель: $model")
    println("Введите сообщение (или 'exit' для выхода)")
    println()

    val history = mutableListOf(
        ChatMessage(role = "system", content = "Ты полезный ассистент. Отвечай кратко и по делу.")
    )

    while (true) {
        print("Вы: ")
        val input = readlnOrNull()?.trim() ?: break

        if (input.equals("exit", ignoreCase = true) || input.equals("quit", ignoreCase = true)) {
            println("До свидания!")
            break
        }

        if (input.isEmpty()) continue

        history.add(ChatMessage(role = "user", content = input))

        try {
            val reply = sendMessage(apiKey, history, model)
            println("\nLLM: $reply\n")
            history.add(ChatMessage(role = "assistant", content = reply))
        } catch (e: Exception) {
            println("\nОшибка: ${e.message}\n")
        }
    }
}
