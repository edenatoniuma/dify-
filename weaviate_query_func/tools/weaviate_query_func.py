import json
import logging
from collections import defaultdict
from collections.abc import Generator
from functools import wraps
from typing import Any, cast
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import weaviate
import requests
from contextlib import contextmanager
import openai
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from weaviate import WeaviateClient
from weaviate.classes.query import Filter, MetadataQuery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)
model_name = "text-embedding-v4"
openai.api_key = "sk-81b101ccde604f25850915286bbd6611"
openai.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

openai_client = OpenAI(api_key=openai.api_key, base_url=openai.base_url)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-api/v1"


@contextmanager
def get_client():
    client = weaviate.connect_to_local(host="192.168.31.33", port=7001, grpc_port=50051)
    try:
        yield client
    finally:
        client.close()


def get_embeddings(text: str) -> CreateEmbeddingResponse:
    embedding_completion = openai_client.embeddings.create(
        model=model_name, input=text, dimensions=1024, encoding_format="float"
    )
    return embedding_completion


def dashscope_rerank(
        query: str,
        documents: list[str],
        top_n: int = 5,
        model: str = "qwen3-rerank",
        instruct: (
                str | None
        ) = "Given a web search query, retrieve relevant passages that answer the query.",
):
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }

    if instruct:
        payload["instruct"] = instruct

    resp = requests.post(
        f"{DASHSCOPE_BASE_URL}/reranks",
        headers={
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def check_collection_exists(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 从参数中取 client 和 collection_name
        client: WeaviateClient | None = kwargs.get("client")
        collection_name: str | None = kwargs.get("collection_name")

        # 也可以根据你自己的风格，从 args 里按位置取
        # client = args[0]; collection_name = args[1] 之类

        if client is None or collection_name is None:
            raise ValueError("需要提供 client 和 collection_name 参数")

        if not client.collections.exists(collection_name):
            logger.error(f"集合：{collection_name} 不存在，无法执行操作")
            return None

        return func(*args, **kwargs)

    return wrapper


@check_collection_exists
def query_data(
        client: WeaviateClient,
        collection_name: str,
        query_text: str,
        custom_return_keys: list[str] | None = None,
        top_k: int = 20,
        rerank_top_k: int = 5,
        doc_name: str | None = None,
) -> dict[str, list[Any]]:
    """
    使用混合检索 +（可选）rerank 查询数据
    """
    collection = client.collections.use(collection_name)

    if not custom_return_keys:
        custom_return_keys = []

    return_keys = ["body", "doc_name", "doc_id", "chunk_index", *custom_return_keys]

    # 1. embedding
    query_vector = get_embeddings(query_text).data[0].embedding

    filters = Filter.by_property(name="doc_name").equal(doc_name) if doc_name else None

    # 2. hybrid retrieve（多取一些，给 rerank 用）
    response = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        limit=top_k,
        return_metadata=MetadataQuery(score=True),
        filters=filters,
    )

    objects = list(response.objects)
    if not objects:
        logger.info("未检索到任何结果")
        return {"result": []}

    # 3. rerank（满足条件才执行）
    reranked_objects = objects

    if 0 < rerank_top_k < len(objects):
        try:
            documents = [cast(str, o.properties.get("body", "")) for o in objects]

            rerank_res = dashscope_rerank(
                query=query_text,
                documents=documents,
                top_n=rerank_top_k,
                instruct="给定一个搜索查询，找出最能回答该查询的相关段落。",
            )

            # 按 rerank index 重排
            reranked_objects = [
                objects[r["index"]] for r in rerank_res.get("results", [])
            ]

            logger.info(f"rerank 生效：{len(objects)} → {len(reranked_objects)}")

        except Exception as e:
            # 降级：保持原排序
            logger.exception("rerank 失败，已降级为原始排序", exc_info=e)
            reranked_objects = objects[:rerank_top_k]

    else:
        # 不 rerank，直接截断
        reranked_objects = objects[:rerank_top_k]

    # 4. 组装返回
    res = defaultdict(list)
    for o in reranked_objects:
        item = {key: o.properties.get(key) for key in return_keys}
        res["result"].append(item)

    logger.info("-------------------------")
    logger.info(f"查询结果为：{json.dumps(res, ensure_ascii=False, indent=2)}")
    logger.info("-------------------------")

    return dict(res)


class WeaviateQueryFuncTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        params = {
            "collection_name": tool_parameters.get("collection_name"),
            "query_text": tool_parameters.get("query_text"),
            "top_k": tool_parameters.get("top_k"),
            "rerank_top_k": tool_parameters.get("rerank_top_k"),
            "doc_name": tool_parameters.get("doc_name")
        }
        with get_client() as client:
            res = query_data(client=client, **params)
        yield self.create_json_message(res)
