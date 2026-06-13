# Sense
**Universal text embedding library for TensorFlow.js and Transformers.js models**
[![npm version](https://img.shields.io/npm/v/@communic/sense?color=%00cc66)](https://www.npmjs.org/package/@communic/sense)

---

Библиотека для получения текстовых эмбеддингов из разных моделей через единый API.

Проект вырос из идеи Universal Sentence Encoder, но не привязан к конкретной архитектуре,
токенизатору или runtime. Он позволяет подключать разные text-embedding backend-ы (векторы
признаков, *feature vectors*), включая TensorFlow.js GraphModel и модели Transformers.js,
например мультиязычные MiniLM-модели.

Основная цель — дать простой и стабильный интерфейс для кодирования строк в dense-векторы,
которые можно использовать в классификации, поиске похожих текстов, кластеризации,
ранжировании и других NLP-задачах.

Любая модель возвращает один и тот же тип результата — `tf.Tensor2D` формы `[N, embeddingDim]`,
поэтому код, который работает с эмбеддингами (классификаторы, поиск похожих, kNN), не зависит от
конкретного рантайма.

> Проект использует часть кода Google «Universal Sentence Encoder» (USE), поэтому
> часть файлов (use_qna.ts и tokenizer.ts) сохраняют оригинальную лицензию Apache 2.0
> Чтобы не путаться с оригиналом от Google, в коде имеются имена с префиксом `Use…`
> (`UseLiteEmbedder`, `UseSentencePieceTokenizer`, `UseQnAEmbedder`) они относятся
> именно к семейству моделей Google USE и сохранены намеренно.


```bash
npm install @communic/sense @tensorflow/tfjs @huggingface/transformers

```

Область применимости намеренно ограничена **моделями текстовых эмбеддингов**.
Это не попытка поддержать «любой transformer вообще».

## Поддерживаемые модели

| `ModelId`                          | Рантайм           | Dim | Токенизация         | Pooling | Normalize | Язык                   |
| ---------------------------------- | ----------------- | --- | ------------------- | ------- | --------- | ---------------------- |
| `use-lite` (по умолчанию)          | `tfjs-graph`      | 512 | USE SentencePiece   | none    | false     | англ.                  |
| `use-qna`                          | `tfjs-graph`      | 100 | USE SentencePiece   | none    | false     | англ. (вопрос-ответ)   |
| `paraphrase-multilingual-minilm`   | `transformers-js` | 384 | внутренняя (runtime)| mean    | true      | мультиязычная (вкл. RU)|

`paraphrase-multilingual-minilm` — это
[`Xenova/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2),
которая поддерживает русский и десятки других языков.

## Архитектура

- **`TextEmbedder`** — единый контракт, который реализует каждый эмбеддер:

  ```ts
  export interface TextEmbedder {
    readonly id: string;
    readonly embeddingDim: number;
    readonly outputType: 'tfjs-tensor2d';
    embed(inputs: string[] | string): Promise<tf.Tensor2D>;
    dispose?(): void;
  }
  ```

- **`ModelProfile`** — статическое описание модели (рантайм, размерность,
  токенизатор, формат входа, pooling, normalize, источники весов). Вместо
  россыпи `modelUrl` / `vocabUrl` / `backend`.
- **`UseLiteEmbedder`** (`tfjs-graph`) — оборачивает классическую USE lite,
  не меняя её численного поведения. Внутри разнесены:
  - `UseSentencePieceTokenizer` — Viterbi-сегментация по словарю
    SentencePiece (бывший `Tokenizer`; это **не** универсальный токенизатор,
    он применим только к USE-профилям);
  - `UseSparseInputAdapter` — строит sparse `{indices, values}`;
  - `TfjsGraphRuntime` — загружает и исполняет `GraphModel`.
- **`TransformersJsEmbedder`** (`transformers-js`) — использует
  feature-extraction pipeline из `@huggingface/transformers`. Токенизация
  скрыта внутри рантайма (`runtime-managed`).
- **`createEmbedder()`** — фабрика, которая по `ModelProfile.runtime`
  собирает нужный эмбеддер.

Токенизаторы описаны через дженерик-адаптер, потому что у разных моделей
разный тип кодирования:

```ts
export interface TokenizerAdapter<TEncoding> {
  encodeBatch(texts: string[]): Promise<TEncoding>;
}
```

## Откуда загружаются веса

Веса USE-моделей **хранятся в этом репозитории** в папке [`models/`](./models)
и по умолчанию грузятся оттуда (через GitHub raw), а не с `tfhub.dev`:

- `models/use-lite/` — `model.json` + шарды + `vocab.json`;
- `models/use-qna/`  — `model.json` + шарды + `vocab.json`.

Источник весов настраивается через `weightsSource`:

| `weightsSource`        | Поведение                                           |
| ---------------------- | --------------------------------------------------- |
| `github` (по умолчанию)| Веса из этого репозитория (GitHub raw).             |
| `tfhub`                | Оригинальные веса с `tfhub.dev` / Kaggle.           |
| `custom`               | Явные `modelUrl` / `vocabUrl`.                      |
| `huggingface`          | Hugging Face Hub (для `transformers-js`).           |

> **Про `paraphrase-multilingual-minilm`:** ONNX-веса модели весят ~118 МБ —
> это больше лимита GitHub на файл (100 МБ), поэтому в репозиторий они **не**
> кладутся. По умолчанию они грузятся с Hugging Face Hub (и кешируются
> `@huggingface/transformers`). Чтобы захостить их самостоятельно (например,
> на своём сервере или в GitHub-репозитории с Git LFS), укажите `remoteHost`
> или `localModelPath`.

## Установка

```bash
npm install sense @tensorflow/tfjs @huggingface/transformers
```

`@huggingface/transformers` — **опциональная** зависимость. Она нужна только
для моделей с рантаймом `transformers-js` и подгружается лениво, поэтому
классический USE-путь не тянет за собой ONNX runtime. Установите её отдельно,
если планируете использовать мультиязычную модель:

```bash
npm install @huggingface/transformers
```

## Использование

```js
require('@tensorflow/tfjs');
const sense = require('@communic/sense');
```

или как отдельный script-тег:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@communic/sense"></script>
```

### USE lite (по умолчанию, 512-dim)

```js
// Поведение полностью совместимо со старой версией.
const model = await sense.load();
const embeddings = await model.embed(['I like my phone.', 'How old are you?']);
// `embeddings` — tf.Tensor2D формы [2, 512].
embeddings.print(true /* verbose */);
```

### Мультиязычная модель (384-dim, поддержка русского)

```js
const model = await sense.load({
  model: 'paraphrase-multilingual-minilm',
});

const embeddings = await model.embed([
  'Оплата по счёту',
  'Банковская карта',
]);
// `embeddings` — tf.Tensor2D формы [2, 384], уже L2-нормализованный.
```

### Опции `load()`

```ts
interface LoadConfig {
  model?: ModelId;              // 'use-lite' | 'use-qna' | 'paraphrase-multilingual-minilm'
  weightsSource?: WeightsSource;// 'github' | 'tfhub' | 'huggingface' | 'custom'

  // tfjs-graph (USE):
  modelUrl?: string;
  vocabUrl?: string;

  // transformers-js (мультиязычная):
  hfModelId?: string;           // переопределить id модели на HF
  remoteHost?: string;          // свой хост для весов (напр. GitHub raw / зеркало)
  localModelPath?: string;      // локальная папка с весами
  dtype?: 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4'; // по умолчанию 'q8'
}
```

### Использование старых USE-весов с tfhub.dev

```js
const model = await sense.load({weightsSource: 'tfhub'});
```

### Токенизатор USE отдельно

```js
const tokenizer = await sense.loadTokenizer();
tokenizer.encode('Привет, как дела?');
```

### USE QnA (двойной энкодер вопрос-ответ)

```js
const model = await sense.loadQnA();
const input = {
  queries: ['Как ты себя чувствуешь сегодня?', 'Какая столица России?'],
  responses: [
    'Я чувствую себя не очень хорошо.',
    'Москва - столица России.',
    'У тебя пять пальцев на руке.',
  ],
};
const embeddings = model.embed(input);
const scores = tf.matMul(
  embeddings['queryEmbedding'], embeddings['responseEmbedding'], false, true
).dataSync();
```

## Версионирование эмбеддинг-бэкенда и головы классификатора

Какой именно эмбеддинг-бэкенд использовался — нужно фиксировать явно, иначе
через месяц будет непонятно, какая голова классификатора к каким эмбеддингам
относится. Храните эти данные **вместе** с весами классификатора:

```ts
interface ClassifierBundleMetadata {
  embeddingModel: ModelId;   // напр. 'paraphrase-multilingual-minilm'
  embeddingDim: number;      // 384
  pooling: 'none' | 'mean';  // 'mean'
  normalize: boolean;        // true
  headVersion: string;       // напр. 'v2'
  labels: string[];
}
```

Удобно собирать метаданные прямо из профиля модели:

```ts
import { MODEL_PROFILES, ClassifierBundleMetadata } from '@communic/sense';

const p = MODEL_PROFILES['paraphrase-multilingual-minilm'];
const meta: ClassifierBundleMetadata = {
  embeddingModel: p.id,
  embeddingDim: p.embeddingDim,
  pooling: p.pooling,
  normalize: p.normalize,
  headVersion: 'v2',
  labels: ['оплата', 'доставка', 'возврат'],
};
```

## Разработка

```bash
npm test     # юнит-тесты (jasmine)
npm run build # сборка dist/sense.{js,min.js,esm.js}
npm run lint  # oxlint
```

## Структура проекта

- `src/`
  - `index.ts` — публичный API (`load`, `loadQnA`, `loadTokenizer`, типы).
  - `types.ts` — `TextEmbedder`, `ModelProfile`, `ModelId`, `LoadConfig`,
    `TokenizerAdapter`, `ClassifierBundleMetadata`.
  - `profiles.ts` — `MODEL_PROFILES`, `resolveProfile()`, источники весов.
  - `factory.ts` — `createEmbedder()`.
  - `embedders/` — `UseLiteEmbedder`, `TransformersJsEmbedder`,
    `UseSparseInputAdapter`.
  - `runtime/` — `TfjsGraphRuntime`.
  - `tokenizer/` — `UseSentencePieceTokenizer` + `Trie`.
  - `use_qna.ts` — `UseQnAEmbedder`.
- `test/` — юнит-тесты (`*_test.ts`) и тестовые фикстуры (`test_util.ts`),
  вынесены отдельно от исходников и не попадают в `dist/`.
- `models/` — предобученные веса USE-моделей (use-lite, use-qna).
- `knn-classifier/` — утилита для классификации (оставлена для обучения).
