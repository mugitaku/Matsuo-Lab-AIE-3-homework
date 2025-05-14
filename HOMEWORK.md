# 1.質問設計の観点と意図
LLMを活用して情報のファクトチェック機能の開発は活発に行われている。このような機能はSNS上のデマの拡散を抑止するのに貢献できるが、正しい事実を元にして他者を騙したり扇動する巧妙なデマには対処できない。そこで、情報に悪意が込められているかどうかをRAGを用いて判断する機能の開発を試みた。

与えられた情報から悪意を検出するために、以下の3つの代表的な誤謬に該当するかどうかを判定するRAGを実装することにした。
*   [衆人に訴える論証
](https://ja.wikipedia.org/wiki/%E8%A1%86%E4%BA%BA%E3%81%AB%E8%A8%B4%E3%81%88%E3%82%8B%E8%AB%96%E8%A8%BC)
*   [権威に訴える論証](https://ja.wikipedia.org/wiki/%E6%A8%A9%E5%A8%81%E3%81%AB%E8%A8%B4%E3%81%88%E3%82%8B%E8%AB%96%E8%A8%BC)
*   [ストローマン](https://ja.wikipedia.org/wiki/%E3%82%B9%E3%83%88%E3%83%AD%E3%83%BC%E3%83%9E%E3%83%B3)

質問は、上記のWikipediaの記事に掲載されている例文から5つ用意した。

1. 「キリスト教信者は世界で最も多いのだから、その教義は真実に違いない。」という主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」
2. 「世界のほとんどの人々が肉を食べているのだから、肉食に関する倫理的問題など存在しない。」という主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」
3. 「アレキサンダー・ポープが言ったように、愛国心は悪党の最後の拠り所だ。」という主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」
4.   X氏「私は子どもが道路で遊ぶのは危険だと思う。」Y氏「そうは思わない。なぜなら子どもが屋外で遊ぶのは良いことだからだ。X氏は子どもを一日中家に閉じ込めておけというが、果たしてそれは正しい子育てなのだろうか。」Y氏の主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」
5.   X氏「私は雨の日が嫌いだ。」Y氏「もし雨が降らなかったら干ばつで農作物は枯れ、ダムは枯渇し我々はみな餓死することになるが、それでもX氏は雨など無くなったほうが良いと言うのであろうか。」Y氏の主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」

正答は以下の通りである。


1.   A.衆人に訴える論証
2.   A.衆人に訴える論証
3.    B.権威に訴える論証
4.    C.ストローマン
5.    C.ストローマン



# 2. RAGの実装方法と工夫点
## 2.1. RAGなし
まずはベースモデルがどの程度知識を持っているか確かめる。

```
messages = [
    {"role": "user", "content": "X氏「私は雨の日が嫌いだ。」Y氏「もし雨が降らなかったら干ばつで農作物は枯れ、ダムは枯渇し我々はみな餓死することになるが、それでもX氏は雨など無くなったほうが良いと言うのであろうか。」Y氏の主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## 2.2. RAGあり
RAGは、第3回講義資料の「2. 文字起こしデータの活用」に記載された実装方法をほぼそのまま利用している。

```
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("infly/inf-retriever-v1-1.5b", trust_remote_code=True)
# In case you want to reduce the maximum length:
emb_model.max_seq_length = 4096
```

[RAGに与えるデータ](https://github.com/mugitaku/Matsuo-Lab-AIE-3-homework/blob/main/Argumentum-ad-populum.txt)は、WIkipediaの記事から取得し、行単位でチャンク化した。
*   [衆人に訴える論証
](https://ja.wikipedia.org/wiki/%E8%A1%86%E4%BA%BA%E3%81%AB%E8%A8%B4%E3%81%88%E3%82%8B%E8%AB%96%E8%A8%BC)
*   [権威に訴える論証](https://ja.wikipedia.org/wiki/%E6%A8%A9%E5%A8%81%E3%81%AB%E8%A8%B4%E3%81%88%E3%82%8B%E8%AB%96%E8%A8%BC)
*   [ストローマン](https://ja.wikipedia.org/wiki/%E3%82%B9%E3%83%88%E3%83%AD%E3%83%BC%E3%83%9E%E3%83%B3)

LLMへの質問文は記事の例文から取得しているので、例文が挙げられているセクションはRAGへ読ませるテキストから除外した。

```
!rm -rf Matsuo-Lab-AIE-3-homework
!git clone https://github.com/mugitaku/Matsuo-Lab-AIE-3-homework
with open("/content/Matsuo-Lab-AIE-3-homework/Argumentum-ad-populum.txt", "r") as f:
  raw_writedown = f.read()
documents = [text.strip() for text in raw_writedown.split("\n")] #行単位でチャンク化

# Retrievalの実行
question = "X氏「私は雨の日が嫌いだ。」Y氏「もし雨が降らなかったら干ばつで農作物は枯れ、ダムは枯渇し我々はみな餓死することになるが、それでもX氏は雨など無くなったほうが良いと言うのであろうか。」Y氏の主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」"

query_embeddings = emb_model.encode([question], prompt_name="query")
document_embeddings = emb_model.encode(documents)

# 各ドキュメントの類似度スコア
scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())
```

```
topk = 5
for i, index in enumerate(scores.argsort()[0][::-1][:topk]):
  print(f"取得したドキュメント{i+1}: (Score: {scores[0][index]})")
  print(documents[index], "\n\n")
```

```
references = "\n".join(["* " + documents[i] for i in scores.argsort()[0][::-1][:topk]])
messages = [
    {"role": "user", "content": f"[参考資料]\n{references}\n\n[質問]X氏「私は雨の日が嫌いだ。」Y氏「もし雨が降らなかったら干ばつで農作物は枯れ、ダムは枯渇し我々はみな餓死することになるが、それでもX氏は雨など無くなったほうが良いと言うのであろうか。」Y氏の主張は次のうちどれに該当するか。「A.衆人に訴える論証」「B.権威に訴える論証」「C.ストローマン」「D.いずれにも該当しない」"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
)
```

```
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

# 3.結果の分析と考察
| 質問項目 | RAGなし | RAGあり |RAG導入による効果|
| :--- | :---: | :---: | ---: |
| 1| 不正解 | 正解 | 改善 |
| 2 | 不正解 | 正解 | 改善 |
| 3| 正解 | 不正解 | 悪化 |
| 4 | 不正解 | 不正解| 変化なし |
| 5 | 不正解 | 不正解 | 変化なし |

衆人に訴える論証においてのみ改善が見られ、権威に訴える論証については悪化した。

RAG導入によって悪化した質問3では、RAGなしの回答は「正解は B. 権威に訴える論証 です。この主張は、権威（アレキサンダー・ポープ）によって支持されている考え方を表明しています。権威が「愛国心は悪党の最後の拠り所」と述べているので、この主張は権威の意見を支持するものです。」と的を射ている。これに対してRAGありの回答は「正解は C. ストローマン です。ストローマンは、相手の意見を歪めて反論する論法です。この主張は、愛国心は悪党の最後の拠り所であるという意見を、相手が主張している内容を歪めて引用し、さらに反論しています。」と、質問文の構造を無視してストローマンの説明を無理やり当てはめている。

今回は講義資料のコードをほとんど変えずに利用したため、RAGのソースから質問文と関連の強いチャンクを5つ抽出して利用した。このソースに含まれる文はどれも質問文と関連性が強いため、抽出するプロセスを削除した方が性能が上がったかもしれない。
