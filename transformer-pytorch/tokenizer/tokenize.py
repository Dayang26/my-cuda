"""
基于 SentencePiece 的分词器训练模块。

SentencePiece 是一种与语言无关的分词工具,它会把文本切分成子词(subword),
无需预先做分词处理;切分算法由 model_type 指定(如 bpe / unigram 等)。

训练成功后,会在 model_name 指定的路径前缀下生成两个文件:
    - {model_name}.model : 分词器模型(二进制),之后用
                           spm.SentencePieceProcessor(model_file=...) 加载,
                           用于文本的编码(encode)/解码(decode)
    - {model_name}.vocab : 词表文件(纯文本),记录每个 token 及其对应的分数
"""

import sentencepiece as spm


# ---------------------------------------------------------------------------
# 旧版写法(已被下方 train 函数取代,此处仅作保留参考):
#
# 早期版本的 SentencePiece 只支持拼接命令行字符串的方式传参,形如:
#     spm.SentencePieceTrainer("--input=xxx --vocab_size=8000 ...")
# 旧写法存在几个问题:
#   1) 参数拼写错误:"vocal_size" 应为 "vocab_size","charactor_coverage"
#      应为 "character_coverage";
#   2) 各段字符串之间缺少空格,拼接后参数会粘连在一起;
#   3) 用 %s 占位符拼命令字符串,可读性和安全性都不如关键字参数。
#
# 新版本推荐直接使用关键字参数调用(见下方 train 函数),更清晰、不易出错。
# ---------------------------------------------------------------------------
# def train(input_file, vocab_size, model_name, model_type, charactor_coverage):

#     input_argument = (
#         "--input=%s"
#         "--vocal_size=%s"
#         "--model_prefix=%s"
#         "--model_type=%s"
#         "--charactor_coverage=%s"
#         "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
#     )

#     cmd = input_argument % (
#         input_file,
#         vocab_size,
#         model_name,
#         model_type,
#         charactor_coverage,
#     )
#     spm.SentencePieceTrainer(cmd)


def train(input_file, vocab_size, model_name, model_type, character_coverage=0.9995):
    """训练一个 SentencePiece 分词器模型。

    Args:
        input_file (str): 训练语料文件路径(纯文本,建议每行一个句子)。
        vocab_size (int): 目标词表大小,即最终训练出的 token 数量上限。
        model_name (str): 模型输出的文件名前缀;训练完成后生成
            {model_name}.model(模型)和 {model_name}.vocab(词表)两个文件。
        model_type (str): 子词切分算法,常用取值:
            - "bpe"     : Byte Pair Encoding,字节对编码
            - "unigram" : Unigram 语言模型,SentencePiece 默认推荐
            - "char"    : 按字符切分
            - "word"    : 按空格预分词后的词切分
        character_coverage (float, optional): 字符覆盖率,表示词表中至少要覆盖
            语料里出现频率累计达到该比例的字符,其余极少数字符会被映射为
            <unk>。日语/中文等多字符集语言一般取 0.9995(默认值),
            英文等拉丁字母语言可设为 1.0。

    特殊 token 及其固定 id(供模型训练时对齐 embedding / padding / label 用):
        id=0 -> <pad> : 填充符,把变长序列补齐到同一批长度
        id=1 -> <unk> : 未知词,表示词表之外的字符/子词
        id=2 -> <bos> : 句子开始符(Begin Of Sequence)
        id=3 -> <eos> : 句子结束符(End Of Sequence)
    """
    spm.SentencePieceTrainer.train(
        input=input_file,  # 训练语料文件路径
        vocab_size=vocab_size,  # 词表大小
        model_prefix=model_name,  # 输出文件前缀(生成 .model 和 .vocab)
        model_type=model_type,  # 切分算法:bpe / unigram / char / word
        character_coverage=character_coverage,  # 字符覆盖率(默认 0.9995)
        pad_id=0,  # <pad> 填充符的 id
        unk_id=1,  # <unk> 未知词的 id
        bos_id=2,  # <bos> 句子开始符的 id
        eos_id=3,  # <eos> 句子结束符的 id
    )


def run():
    en_input = "transformer-pytorch/data/corpus.en"
    en_vacab_size = 32000
    en_model_name = "eng_spm_model"
    en_model_type = "bpe"
    en_charactor_coverage = 1.0

    train(en_input, en_vacab_size, en_model_name, en_model_type, en_charactor_coverage)

    ch_input = "transformer-pytorch/data/corpus.ch"
    ch_vacab_size = 32000
    ch_model_name = "ch_spm_model"
    ch_model_type = "bpe"
    ch_charactor_coverage = 0.9995

    train(ch_input, ch_vacab_size, ch_model_name, ch_model_type, ch_charactor_coverage)


def test():
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今天抵达夏威夷"

    sp.Load("./ch_spm_model.model")

    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))

    a = [19207, 277, 7419, 7318, 18384, 28724]
    print(sp.DecodeIds(a))


if __name__ == "__main__":
    run()

    # test()
