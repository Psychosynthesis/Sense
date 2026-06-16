# Sense
[![npm version](https://img.shields.io/npm/v/@communic/sense?color=00cc66)](https://www.npmjs.org/package/@communic/sense)

**Universal text embedding library for TensorFlow.js and Transformers.js models**

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

Область применимости намеренно ограничена **моделями текстовых эмбеддингов**.
Это не попытка поддержать «любой transformer вообще», это расширение USE на модели для других языков.

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

Код примера:

```ts
 // Точно воспроизвести поведение Google USE (грузить с tfhub):
    const embedder = await sense.load({ weightsSource: 'tfhub' });
 // Дефолтный 'github' грузит из репозитория
 // Указать свои URL (self-host):
    const embedder = await sense.load({
      modelUrl: 'https://your-host/use-lite/model.json',
      vocabUrl: 'https://your-host/use-lite/vocab.json',
    });
```

> **Про `paraphrase-multilingual-minilm`:** ONNX-веса модели весят ~118 МБ —
> это больше лимита GitHub на файл (100 МБ), поэтому в репозиторий они **не**
> кладутся. По умолчанию они грузятся с Hugging Face Hub (и кешируются
> `@huggingface/transformers`). Чтобы захостить их самостоятельно (например,
> на своём сервере или в GitHub-репозитории с Git LFS), укажите `remoteHost`
> или `localModelPath`.

## Установка

```bash
npm install @communic/sense @tensorflow/tfjs @huggingface/transformers
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
const model = await sense.load({ weightsSource: 'tfhub' });
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


## Терминология
**Embedding (эмбеддинг)** — это числовое представление текста в виде вектора фиксированной длины (вектора признаков). Модель превращает произвольную строку в массив чисел, в котором семантически близкие тексты оказываются рядом в пространстве, а далёкие — далеко. В этой библиотеке эмбеддинг — это всегда результат метода embed(), то есть tf.Tensor2D формы [N, embeddingDim], где N — число входных строк, а embeddingDim — размерность модели (512 у use-lite, 384 у paraphrase-multilingual-minilm). Размерность и прочие свойства фиксированы в ModelProfile (embeddingDim, pooling, normalize).
Именно ради единообразия эмбеддингов существует общий контракт TextEmbedder: что бы ни стояло под капотом (USE или мультиязычный трансформер), на выходе всегда один и тот же тип — tf.Tensor2D. Это позволяет писать код «над эмбеддингами» (поиск похожих, кластеризацию, классификатор), не привязываясь к конкретной модели. У мультиязычной модели эмбеддинги ещё и L2-нормализованы (normalize: true), то есть их длина равна 1 — это удобно для сравнения через косинусную близость / скалярное произведение.

**Head («голова»)** — это отдельная обучаемая модель, которая надстраивается поверх эмбеддингов и решает конкретную прикладную задачу (например, категоризацию новостей). Сама эта библиотека голову не содержит и не обучает — она лишь поставляет эмбеддинги (это «тело»/backbone). Голова — это, как правило, маленькая нейросеть (несколько dense-слоёв с финальным sigmoid/softmax), которая на вход получает вектор эмбеддинга (иногда сконкатенированный с другими признаками вроде TF-IDF) и выдаёт вероятности классов.
Ключевой момент: голова жёстко завязана на тот эмбеддинг-бэкенд, на котором её обучали. Если сменить модель эмбеддингов (например, use-lite 512-dim → paraphrase-multilingual-minilm 384-dim), старая голова станет несовместимой — у неё не сойдётся размерность входа, да и само пространство признаков будет другим. Поэтому в репе есть тип ClassifierBundleMetadata (embeddingModel, embeddingDim, pooling, normalize, headVersion, labels): он хранится рядом с весами головы и однозначно фиксирует, к каким именно эмбеддингам эта голова относится.

**Sparse («разреженный»)** — это формат входа для классической модели USE lite, обозначенный в профиле как inputFormat: 'use-sparse'. Вместо плотной матрицы токенов модель ожидает разреженное представление: пару тензоров indices (матрица [nnz, 2] из пар [номер строки, позиция токена]) и values (вектор [nnz] с самими id токенов), где nnz — суммарное число токенов во всём батче. То есть хранятся только «ненулевые» элементы и их координаты, а не вся прямоугольная матрица с паддингом.
За построение этого формата отвечает UseSparseInputAdapter: он берёт токенизированные строки (после UseSentencePieceTokenizer) и собирает {indices, values}, которые затем скармливаются GraphModel. Эта логика раньше была «вшита» прямо в embed(), а теперь вынесена в отдельный адаптер. У мультиязычной модели sparse-вход не используется вовсе — там токенизация и формат входа полностью спрятаны внутри рантайма (inputFormat: 'runtime-managed').

**Backend / Runtime (бэкенд)** — в данном проекте слово «бэкенд» имеет два смысла, и их полезно различать. Первый — это runtime модели, поле ModelProfile.runtime, которое говорит, чем исполняется модель: tfjs-graph (это TensorFlow.js GraphModel, путь USE — обёрнут в TfjsGraphRuntime) или transformers-js (ONNX-пайплайн @huggingface/transformers, путь мультиязычной модели). Фабрика createEmbedder() по этому полю выбирает, какой эмбеддер собрать. Именно это чаще всего имеется в виду под «эмбеддинг-бэкендом» (на каком движке/модели получены эмбеддинги).
Второй смысл — это вычислительный backend самого TensorFlow.js: cpu, tensorflow (через @tensorflow/tfjs-node), webgl и т.д. Он определяет, на чём физически считаются тензорные операции (CPU/GPU). В тестах, например, явно ставится CPU-бэкенд (tf.setBackend('cpu')). Этот низкоуровневый backend ортогонален к runtime из профиля: он влияет на скорость вычислений, но не на то, какая модель и каким способом строит эмбеддинги.


## Примеры
 - [Categorizer](https://github.com/Psychosynthesis/Categorizer): пример кода обучающего небольшую нейросеть-категоризатор

## Какие зависимости нужны в проектах-потребителях
  `@tensorflow/tfjs-node` — нужен везде. Он закрывает сразу три вещи:

  >Peer-зависимости Sense. Библиотека объявляет peer-deps @tensorflow/tfjs-core и @tensorflow/tfjs-converter (через converter грузится USE GraphModel). @tensorflow/tfjs-node тянет за собой полный @tensorflow/tfjs (core + converter + layers + …), поэтому оба peer-deps удовлетворяются транзитивно — отдельно ставить tfjs-core/tfjs-converter не обязательно (хотя можно, чтобы убрать возможные peerDep-варнинги npm).

  >file://-загрузчик. Ты грузишь голову через tf.loadLayersModel('file://...'). IO-обработчик схемы file:// в Node предоставляет именно @tensorflow/tfjs-node; у чистого @tensorflow/tfjs его нет. Без tfjs-node загрузка головы с диска не заработает.

  >Нативный backend — быстрые вычисления predict/concat/tensor2d на CPU.

 **Важно**: версия `tfjs-node` должна попадать в диапазон peer-deps Sense (^4.22.0), иначе будет дубликат tfjs-core и конфликт реестров бэкендов.

`@huggingface/transformers` — по ситуации. Эта зависимость нужна только для рантайма `transformers-js`, то есть когда `loadModel({ model })` получает `paraphrase-multilingual-minilm`. Импорт её в библиотеке ленивый: для моделей `use-lite/use-qna` (рантайм `tfjs-graph`) она вообще не подгружается, и ставить её не надо. Если же голова обучалась на мультиязычной модели — она обязательна (вместе с ней подтянется onnxruntime-node), иначе при `embedder.embed()` будет явная ошибка вида `«requires the optional dependency @huggingface/transformers»`.

**Нюанс про установку**: в **Sense** `@huggingface/transformers` указан как `optionalDependencies`, а такие зависимости npm по умолчанию ставит автоматически. То есть после обычного `npm install @communic/sense` она, скорее всего, уже окажется в `node_modules` сама — отдельно ставить не придётся. Явно поставить её нужно, только если (а) модель мультиязычная и (б) установка была с `--omit=optional` или установка optional-зависимости упала (например, не собрался `onnxruntime` под платформу). И наоборот: если у тебя голова только на USE и ты хочешь не тащить тяжёлый `onnxruntime`, ставь зависимости с `--omit=optional`.
