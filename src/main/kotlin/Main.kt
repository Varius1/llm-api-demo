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
data class TokenUsage(
    @SerialName("prompt_tokens") val promptTokens: Int = 0,
    @SerialName("completion_tokens") val completionTokens: Int = 0,
    @SerialName("total_tokens") val totalTokens: Int = 0,
)

@Serializable
data class ChatResponse(
    val choices: List<ChatChoice> = emptyList(),
    val error: ChatError? = null,
    val usage: TokenUsage? = null,
    val model: String? = null,
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
    .readTimeout(120, TimeUnit.SECONDS)
    .build()

// --- Конфигурация моделей для бенчмарка ---

data class ModelConfig(
    val id: String,
    val tier: String,
    val displayName: String,
    val inputPricePerMillion: Double,
    val outputPricePerMillion: Double,
    val url: String,
)

private val BENCHMARK_MODELS = listOf(
    ModelConfig(
        id = "meta-llama/llama-3.3-70b-instruct",
        tier = "Слабая (дешёвая)",
        displayName = "Llama 3.3 70B",
        inputPricePerMillion = 0.10,
        outputPricePerMillion = 0.32,
        url = "https://openrouter.ai/meta-llama/llama-3.3-70b-instruct",
    ),
    ModelConfig(
        id = "google/gemini-2.5-flash",
        tier = "Средняя",
        displayName = "Gemini 2.5 Flash",
        inputPricePerMillion = 0.30,
        outputPricePerMillion = 2.50,
        url = "https://openrouter.ai/google/gemini-2.5-flash",
    ),
    ModelConfig(
        id = "anthropic/claude-sonnet-4",
        tier = "Сильная (дорогая)",
        displayName = "Claude Sonnet 4",
        inputPricePerMillion = 3.00,
        outputPricePerMillion = 15.00,
        url = "https://openrouter.ai/anthropic/claude-sonnet-4",
    ),
)

private const val BENCHMARK_PROMPT =
    "Объясни принцип работы квантового компьютера и в чём его отличие от классического. Ответ дай в 3-5 предложениях."

private val JUDGE_MODEL = ModelConfig(
    id = "google/gemini-2.5-flash",
    tier = "Судья",
    displayName = "Gemini 2.5 Flash",
    inputPricePerMillion = 0.30,
    outputPricePerMillion = 2.50,
    url = "https://openrouter.ai/google/gemini-2.5-flash",
)

// --- Результат запроса с метриками ---

data class BenchmarkResult(
    val model: ModelConfig,
    val response: String,
    val usage: TokenUsage?,
    val durationMs: Long,
    val costUsd: Double,
    val error: String? = null,
)

// --- Основная логика ---

fun sendMessageRaw(apiKey: String, messages: List<ChatMessage>, model: String, temperature: Double?): ChatResponse {
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

        return json.decodeFromString<ChatResponse>(body)
    }
}

fun sendMessage(apiKey: String, messages: List<ChatMessage>, model: String, temperature: Double?): String {
    val chatResponse = sendMessageRaw(apiKey, messages, model, temperature)

    if (chatResponse.error != null) {
        error("API ошибка: ${chatResponse.error.message}")
    }

    return chatResponse.choices.firstOrNull()?.message?.content
        ?: error("Нет ответа в choices")
}

fun calculateCost(usage: TokenUsage?, model: ModelConfig): Double {
    if (usage == null) return 0.0
    val inputCost = usage.promptTokens.toDouble() / 1_000_000 * model.inputPricePerMillion
    val outputCost = usage.completionTokens.toDouble() / 1_000_000 * model.outputPricePerMillion
    return inputCost + outputCost
}

fun runBenchmark(apiKey: String, prompt: String, models: List<ModelConfig>, temperature: Double?) {
    val messages = listOf(ChatMessage(role = "user", content = prompt))
    val results = mutableListOf<BenchmarkResult>()

    println("╔══════════════════════════════════════════════════════════════╗")
    println("║              СРАВНЕНИЕ МОДЕЛЕЙ OpenRouter                   ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
    println("Промпт: \"$prompt\"")
    println("Температура: ${temperature ?: "по умолчанию"}")
    println()

    for (model in models) {
        println("━".repeat(62))
        println("▶ ${model.tier}: ${model.displayName}")
        println("  ID: ${model.id}")
        println("  Цена: \$${model.inputPricePerMillion}/M вход | \$${model.outputPricePerMillion}/M выход")
        println()

        try {
            val startTime = System.nanoTime()
            val chatResponse = sendMessageRaw(apiKey, messages, model.id, temperature)
            val durationMs = (System.nanoTime() - startTime) / 1_000_000

            if (chatResponse.error != null) {
                val result = BenchmarkResult(model, "", null, durationMs, 0.0, chatResponse.error.message)
                results.add(result)
                println("  ❌ Ошибка: ${chatResponse.error.message}")
                println()
                continue
            }

            val content = chatResponse.choices.firstOrNull()?.message?.content ?: "(пустой ответ)"
            val usage = chatResponse.usage
            val cost = calculateCost(usage, model)

            val result = BenchmarkResult(model, content, usage, durationMs, cost)
            results.add(result)

            println("  Ответ:")
            content.lines().forEach { println("    $it") }
            println()
            println("  ⏱  Время: ${durationMs}мс (${String.format("%.1f", durationMs / 1000.0)}с)")
            if (usage != null) {
                println("  📊 Токены: ${usage.promptTokens} вход + ${usage.completionTokens} выход = ${usage.totalTokens} всего")
            }
            println("  💰 Стоимость: \$${String.format("%.6f", cost)}")
        } catch (e: Exception) {
            val result = BenchmarkResult(model, "", null, 0, 0.0, e.message)
            results.add(result)
            println("  ❌ Ошибка: ${e.message}")
        }
        println()
    }

    printComparison(results)
    runJudge(apiKey, prompt, results)
}

fun printComparison(results: List<BenchmarkResult>) {
    val successful = results.filter { it.error == null }
    if (successful.isEmpty()) {
        println("Нет успешных результатов для сравнения.")
        return
    }

    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                    СВОДНАЯ ТАБЛИЦА                         ║")
    println("╠══════════════════════════════════════════════════════════════╣")

    val header = String.format(
        "║ %-18s │ %8s │ %6s │ %6s │ %10s ║",
        "Модель", "Время", "Вход", "Выход", "Стоимость"
    )
    println(header)
    println("╠══════════════════════════════════════════════════════════════╣")

    for (r in results) {
        val timeStr = if (r.error != null) "ОШИБКА" else "${String.format("%.1f", r.durationMs / 1000.0)}с"
        val inTokens = r.usage?.promptTokens?.toString() ?: "-"
        val outTokens = r.usage?.completionTokens?.toString() ?: "-"
        val costStr = if (r.error != null) "-" else "\$${String.format("%.6f", r.costUsd)}"
        val name = r.model.displayName.take(18)

        println(
            String.format(
                "║ %-18s │ %8s │ %6s │ %6s │ %10s ║",
                name, timeStr, inTokens, outTokens, costStr
            )
        )
    }

    println("╚══════════════════════════════════════════════════════════════╝")
    println()

    val fastest = successful.minByOrNull { it.durationMs }
    val cheapest = successful.minByOrNull { it.costUsd }
    val longest = successful.maxByOrNull { it.response.length }

    println("📈 ВЫВОДЫ:")
    println("─".repeat(62))
    if (fastest != null)
        println("  🚀 Быстрее всех: ${fastest.model.displayName} (${String.format("%.1f", fastest.durationMs / 1000.0)}с)")
    if (cheapest != null)
        println("  💰 Дешевле всех: ${cheapest.model.displayName} (\$${String.format("%.6f", cheapest.costUsd)})")
    if (longest != null)
        println("  📝 Самый подробный: ${longest.model.displayName} (${longest.response.length} символов)")
    println()

    println("🔗 Ссылки на модели:")
    for (r in results) {
        println("  • ${r.model.displayName}: ${r.model.url}")
    }
    println()
}

fun runJudge(apiKey: String, originalPrompt: String, results: List<BenchmarkResult>) {
    val successful = results.filter { it.error == null }
    if (successful.size < 2) return

    println("╔══════════════════════════════════════════════════════════════╗")
    println("║            🧑‍⚖️ ОЦЕНКА КАЧЕСТВА (модель-судья)               ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println("  Судья: ${JUDGE_MODEL.displayName} (${JUDGE_MODEL.id})")
    println()

    val answersBlock = successful.joinToString("\n\n") { r ->
        "--- ${r.model.displayName} (${r.model.tier}) ---\n${r.response}"
    }

    val judgePrompt = """
Ты — эксперт-оценщик ответов языковых моделей. Тебе дан один и тот же вопрос и ответы от ${successful.size} разных моделей.

Исходный вопрос: "$originalPrompt"

Ответы моделей:
$answersBlock

Оцени каждый ответ по критериям:
1. Точность (фактическая правильность)
2. Полнота (насколько полно раскрыта тема)
3. Качество языка (грамотность, нет ли артефактов, мусорных символов, смешения языков)
4. Следование инструкции (уложился ли в 3-5 предложений)

Дай краткую оценку каждой модели (1-2 предложения) и назови лучший ответ. Отвечай на русском.
""".trim()

    try {
        val startTime = System.nanoTime()
        val messages = listOf(ChatMessage(role = "user", content = judgePrompt))
        val response = sendMessageRaw(apiKey, messages, JUDGE_MODEL.id, 0.3)
        val durationMs = (System.nanoTime() - startTime) / 1_000_000

        if (response.error != null) {
            println("  ❌ Ошибка судьи: ${response.error.message}")
            return
        }

        val verdict = response.choices.firstOrNull()?.message?.content ?: "(пустой ответ)"
        val usage = response.usage
        val cost = calculateCost(usage, JUDGE_MODEL)

        println("  Вердикт:")
        verdict.lines().forEach { println("    $it") }
        println()
        println("  ⏱  Время оценки: ${durationMs}мс (${String.format("%.1f", durationMs / 1000.0)}с)")
        if (usage != null) {
            println("  📊 Токены судьи: ${usage.promptTokens} вход + ${usage.completionTokens} выход = ${usage.totalTokens}")
        }
        println("  💰 Стоимость оценки: \$${String.format("%.6f", cost)}")
        println()
    } catch (e: Exception) {
        println("  ❌ Ошибка при вызове судьи: ${e.message}")
        println()
    }
}

fun main(args: Array<String>) {
    val apiKey = System.getenv("OPENROUTER_API_KEY")
        ?: run {
            print("Введите ваш OpenRouter API ключ: ")
            readlnOrNull()?.trim() ?: error("API ключ не введён")
        }

    if (args.contains("--compare")) {
        val customPrompt = args.indexOf("--prompt").let { idx ->
            if (idx >= 0 && idx + 1 < args.size) args[idx + 1] else null
        }
        val temperature = args.indexOf("--temp").let { idx ->
            if (idx >= 0 && idx + 1 < args.size) args[idx + 1].toDoubleOrNull() else null
        }

        runBenchmark(apiKey, customPrompt ?: BENCHMARK_PROMPT, BENCHMARK_MODELS, temperature)
        return
    }

    val model = System.getenv("LLM_MODEL") ?: DEFAULT_MODEL
    var temperature = System.getenv("LLM_TEMPERATURE")?.toDoubleOrNull()

    println("=== LLM CLI Chat ===")
    println("Модель: $model")
    println("Температура: ${temperature ?: "по умолчанию (1.0)"}")
    println("Команды: /temp 0.7 — температура, /compare — сравнить модели, exit — выход")
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
                if (lines.isEmpty() && trimmed == "/compare") {
                    println()
                    runBenchmark(apiKey, BENCHMARK_PROMPT, BENCHMARK_MODELS, temperature)
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
            val reply = sendMessage(apiKey, messages, model, temperature)
            println("\nLLM:\n$reply\n")
        } catch (e: Exception) {
            println("\nОшибка: ${e.message}\n")
        }
    }
}
