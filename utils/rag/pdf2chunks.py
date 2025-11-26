import re
from typing import Optional, List, Tuple
from langchain_core.documents import Document
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from langdetect import detect
from utils.logger import get_logger

# 设置日志格式
logger = get_logger(__name__)

# pdf文件源存储路径
file_source_path: str = ''


def sent_tokenize(text: str) -> List[str]:
    """
    根据语言对文本进行句子分割。
    Args:
        text (str): 待分割的原始文本。

    Returns:
        List[str]: 分割后的句子列表（已去除首尾空格，过滤空句）。
    """
    try:
        lang = detect(text)  # 自动检测语言
    except:
        lang = "en"  # 检测失败时默认为英文

    if lang.startswith("zh"):
        # 中文句子分割：利用正向后瞻断言，确保在标点后切分，但保留标点
        sents = re.split(r'(?<=[。！？；!?])', text)
    else:
        # 英文：使用更健壮的正则（不依赖 NLTK）
        # 匹配 . ! ? 后跟空格或换行，且后面是大写字母或引号+大写（避免误切如 "Mr."）
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'“])', text)
        # 如果正则没切开（比如全是小写），退回到简单切分
        if len(sents) == 1 and len(text) > 100:
            sents = re.split(r'(?<=[.!?])\s+', text)

    # 清理：去除空字符串和首尾空格
    return [s.strip() for s in sents if s.strip()]


def extract_paragraphs_by_page(
        filename: str,
        page_numbers: Optional[List[int]] = None,
        min_line_length: int = 1
) -> List[Tuple[int, str]]:
    """
    从 PDF 文件中逐页提取段落，并记录每个段落所属的页码（从 0 开始）。

    提取逻辑：
      - 遍历每一页，跳过不在 page_numbers 中的页（若指定）。
      - 对每页内的文本行进行合并，处理连字符换行。
      - 按空行或短行（< min_line_length）作为段落分隔符。

    Args:
        filename (str): PDF 文件路径。
        page_numbers (Optional[List[int]]): 要提取的页码列表（从 0 开始）。若为 None，则提取全部页。
        min_line_length (int): 判定为“有效行”的最小字符长度。短于此长度的行视为段落分隔符。

    Returns:
        List[Tuple[int, str]]: 每个元素为 (页码, 段落文本)。
    """
    page_paragraphs: List[Tuple[int, str]] = []

    # 使用 pdfminer 逐页解析 PDF
    for page_idx, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过非目标页
        if page_numbers is not None and page_idx not in page_numbers:
            continue

        # 收集当前页的所有文本行
        page_text_lines = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text_lines.append(element.get_text())

        # 将当前页的所有文本行合并为一个字符串（保留换行）
        full_page_text = '\n'.join(page_text_lines)
        lines = full_page_text.split('\n')

        # 缓冲区：用于拼接属于同一段落的行
        buffer = ''
        paragraphs = []

        # 遍历每一行，构建段落
        for text in lines:
            text = text.rstrip()  # 去除行尾空白（包括换行符）
            if len(text) >= min_line_length:
                # 有效行：拼接到缓冲区
                if text.endswith('-'):
                    # 行尾是连字符，表示单词被截断，需去掉连字符并直接拼接
                    buffer += text.strip('-')
                else:
                    # 普通行：前面加空格（避免单词粘连）
                    buffer += ' ' + text if buffer else text
            else:
                # 空行或短行：视为段落结束
                if buffer:
                    paragraphs.append(buffer)
                    buffer = ''

        # 处理最后剩余的缓冲区
        if buffer:
            paragraphs.append(buffer)

        # 将当前页的所有段落与页码绑定，加入结果列表
        for para in paragraphs:
            page_paragraphs.append((page_idx, para))

    return page_paragraphs


def split_text_with_page(
        paragraphs_with_page: List[Tuple[int, str]],
        chunk_size: int = 800,
        overlap_size: int = 200
) -> List[Document]:
    """
    将带页码的段落列表切分为具有重叠（overlap）的文本块（chunks），
    每个文本块以完整句子为单位，避免在句子中间切断，
    并保留其来源页码（取第一个句子所在的页）。

    Args:
        paragraphs_with_page: List[Tuple[int, str]]，每个元素为 (页码, 段落文本)
        chunk_size: 每个 chunk 的目标最大字符长度（包含空格）
        overlap_size: 相邻 chunks 之间的最小重叠字符数

    Returns:
        List[Document]：每个 Document 包含 page_content 和 metadata（含 page）
    """
    # 第一步：将所有段落按句子粒度展开，并保留每个句子对应的页码
    sentences_with_page: List[Tuple[int, str]] = []
    for page_idx, para in paragraphs_with_page:
        # 对当前段落进行句子分割
        sents = sent_tokenize(para)
        for sent in sents:
            # 过滤掉空句子（防止因 tokenize 引入空字符串）
            if sent.strip():
                sentences_with_page.append((page_idx, sent.strip()))

    # 如果没有有效句子，直接返回空列表
    if not sentences_with_page:
        return []

    # 初始化结果列表和指针
    chunks: List[Document] = []
    i = 0  # 当前处理到的句子索引
    total = len(sentences_with_page)
    global file_source_path

    # 第二步：滑动窗口式构建 chunks，确保重叠
    while i < total:
        # 记录当前 chunk 的起始位置（用于后续计算重叠）
        start_i = i
        # 当前 chunk 的起始页码（取第一个句子的页码）
        current_page = sentences_with_page[i][0]
        # 初始化当前 chunk 的文本内容为第一个句子
        current_chunk = sentences_with_page[i][1]
        i += 1  # 移动到下一个句子

        # 尽可能向后扩展当前 chunk，直到加上下一个句子会超过 chunk_size
        while i < total and len(current_chunk) + len(sentences_with_page[i][1]) + 1 <= chunk_size:
            # +1 是为了预留一个空格（句子间用空格连接）
            current_chunk += ' ' + sentences_with_page[i][1]
            i += 1

        # 构造 LangChain 的 Document 对象
        doc = Document(
            page_content=current_chunk,
            metadata={"page": current_page, "source": file_source_path}
        )
        chunks.append(doc)

        # 第三步：计算下一个 chunk 的起始位置，以实现 overlap
        # 如果已经处理完所有句子，跳出循环
        if i >= total:
            break

        # 从当前 chunk 的末尾开始向前回溯，累计字符数，直到达到 overlap_size
        overlap_chars = 0
        overlap_start_idx = i - 1  # 从当前 chunk 的最后一个句子开始往前找

        # 向前遍历，直到累计重叠字符数 >= overlap_size 或回到当前 chunk 起始位置
        while overlap_start_idx >= start_i and overlap_chars < overlap_size:
            sent_text = sentences_with_page[overlap_start_idx][1]
            overlap_chars += len(sent_text) + 1  # +1 代表句子间的空格
            overlap_start_idx -= 1

        # 确定下一个 chunk 的起始索引：
        # overlap_start_idx 是最后一个“未被包含”的句子索引，
        # 所以下一个 chunk 应从 overlap_start_idx + 1 开始
        next_start = overlap_start_idx + 1

        # 安全兜底：避免 next_start <= start_i 导致死循环或重复
        # 至少前进一个句子
        if next_start <= start_i:
            next_start = start_i + 1

        # 如果 next_start 超出范围，结束循环
        if next_start >= total:
            break

        # 更新 i，开始构建下一个 chunk
        i = next_start

    return chunks


def pdf_to_chunks(
        file_path: str,
        page_numbers: Optional[List[int]] = None,
        min_line_length: int = 1
) -> List[Document]:
    """
    主入口函数：从 PDF 文件加载内容，切分为 LangChain 兼容的 Document 列表。

    流程：
      1. 按页提取段落（保留页码）。
      2. 将段落拆分为句子。
      3. 合并句子为不超过 chunk_size 的文本块。
      4. 为每个块创建 Document，填充 source 和 page 元数据。

    Args:
        file_path (str): PDF 文件路径。
        page_numbers (Optional[List[int]]): 要处理的页码列表（从 0 开始），None 表示全部页。
        min_line_length (int): 用于段落分割的最小行长度阈值。

    Returns:
        List[Document]: 切分后的文档列表，可直接用于 LangChain 向量库。
    """

    global file_source_path
    file_source_path = file_path
    # 步骤 1：提取带页码的段落
    paragraphs_with_page = extract_paragraphs_by_page(
        filename=file_path,
        page_numbers=page_numbers,
        min_line_length=min_line_length
    )
    if not paragraphs_with_page:
        logger.warning(f"未从 {file_path} 提取到文本（可能为扫描件）")
        return []

    # 步骤 2：切分为带元数据的 Document chunks
    chunks = split_text_with_page(
        paragraphs_with_page=paragraphs_with_page,
        chunk_size=800,
        overlap_size=200
    )

    logger.info(f"从文件 '{file_path}' 成功提取 {len(chunks)} 个文本块（chunks）")
    return chunks


# test
if __name__ == "__main__":
    chunks = pdf_to_chunks(
        "../../document/健康档案.pdf",
        # page_numbers=[2, 3], # 指定页面
        page_numbers=None,  # 加载全部页面
        min_line_length=1
    )
    # 测试前3条文本
    logger.info(f"只展示3段截取片段:")
    logger.info(f"截取的片段1: {chunks[0]}")
    logger.info(f"截取的片段2: {chunks[2]}")
    logger.info(f"截取的片段3: {chunks[3]}")
