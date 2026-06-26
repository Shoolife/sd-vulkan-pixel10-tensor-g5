package com.example.generet_image_ai.sd

import android.content.Context
import org.json.JSONObject

/**
 * CLIP byte-level BPE токенизатор (порт openai/CLIP simple_tokenizer) для SD1.5.
 * Превращает произвольный текст промпта в int32[77] токены для clip.tflite.
 *
 * Ресурсы в assets/sd/: vocab.json (token→id), merges.txt (BPE-ранги).
 */
class ClipTokenizer(context: Context) {
    private val encoder: Map<String, Int>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val byteEncoder: Map<Int, Char>
    private val cache = HashMap<String, List<String>>()

    private val bos = 49406
    private val eos = 49407
    private val maxLen = 77

    // CLIP-разбиение на токены-слова (contractions, буквы, цифры, прочее)
    private val pat = Regex(
        "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|\\p{L}+|\\p{N}|[^\\s\\p{L}\\p{N}]+",
        RegexOption.IGNORE_CASE,
    )

    init {
        // vocab.json → encoder
        val vocabTxt = context.assets.open("sd/vocab.json").bufferedReader().use { it.readText() }
        val jo = JSONObject(vocabTxt)
        val enc = HashMap<String, Int>(jo.length() * 2)
        val keys = jo.keys()
        while (keys.hasNext()) { val k = keys.next(); enc[k] = jo.getInt(k) }
        encoder = enc

        // merges.txt → ранги пар (строка N = ранг N)
        val ranks = HashMap<Pair<String, String>, Int>()
        context.assets.open("sd/merges.txt").bufferedReader().useLines { lines ->
            var rank = 0
            for (line in lines) {
                if (line.startsWith("#") || line.isBlank()) continue
                val sp = line.split(' ')
                if (sp.size == 2) ranks[sp[0] to sp[1]] = rank++
            }
        }
        bpeRanks = ranks
        byteEncoder = bytesToUnicode()
    }

    /** Байт→печатный unicode-символ (CLIP byte-level), чтобы пробелы/служебные байты были видимы. */
    private fun bytesToUnicode(): Map<Int, Char> {
        val bs = ArrayList<Int>()
        bs.addAll(33..126); bs.addAll(161..172); bs.addAll(174..255)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        return bs.indices.associate { bs[it] to cs[it].toChar() }
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = HashSet<Pair<String, String>>()
        for (i in 0 until word.size - 1) pairs.add(word[i] to word[i + 1])
        return pairs
    }

    /** BPE-слияние одного токена-слова в подтокены по рангам merges. */
    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }
        // символы слова; к последнему добавляем </w>
        var word = token.map { it.toString() }.toMutableList()
        if (word.isEmpty()) return emptyList()
        word[word.size - 1] = word.last() + "</w>"

        while (true) {
            val pairs = getPairs(word)
            if (pairs.isEmpty()) break
            val bigram = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (bpeRanks[bigram] == null) break
            val (first, second) = bigram
            val newWord = ArrayList<String>()
            var i = 0
            while (i < word.size) {
                var j = i
                while (j < word.size && word[j] != first) j++
                if (j >= word.size) { newWord.addAll(word.subList(i, word.size)); break }
                newWord.addAll(word.subList(i, j))
                if (j < word.size - 1 && word[j + 1] == second) {
                    newWord.add(first + second); i = j + 2
                } else { newWord.add(word[j]); i = j + 1 }
            }
            word = newWord
            if (word.size == 1) break
        }
        cache[token] = word
        return word
    }

    /** Текст → int32[77]: BOS + токены (обрезка до 75) + EOS, паддинг EOS до 77. */
    fun encode(text: String): IntArray {
        val clean = text.trim().lowercase().replace(Regex("\\s+"), " ")
        val ids = ArrayList<Int>()
        ids.add(bos)
        for (m in pat.findAll(clean)) {
            val tok = m.value
            // байты UTF-8 → byte-level unicode строка
            val sb = StringBuilder()
            for (b in tok.toByteArray(Charsets.UTF_8)) sb.append(byteEncoder[b.toInt() and 0xFF])
            for (sub in bpe(sb.toString())) {
                encoder[sub]?.let { ids.add(it) }
            }
        }
        if (ids.size > maxLen - 1) { // оставить место под EOS
            while (ids.size > maxLen - 1) ids.removeAt(ids.size - 1)
        }
        ids.add(eos)
        val out = IntArray(maxLen) { eos }
        for (i in ids.indices) out[i] = ids[i]
        return out
    }
}
