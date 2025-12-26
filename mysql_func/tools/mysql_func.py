import json
import logging
from collections.abc import Generator
from typing import Any
import pymysql
from pymysql.cursors import DictCursor
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class MysqlFuncTool(Tool):
    """
    输入MySQL数据库的连接信息，获取对应的数据以及执行SQL语句获取对应的结果
    """

    FORBIDDEN_SQL_KEYWORDS = {
        "UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "REPLACE", "RENAME", "GRANT", "REVOKE", "SET",
        "CALL", "USE", "MERGE"
    }

    ALLOWED_SQL_PREFIXES = {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN"}

    def _clean_sql_prefix(self, sql: str) -> str:
        """
        去除前置注释与空格，返回第一个关键词
        """
        s = sql.strip()

        # 去除块注释 /* ... */
        while s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                break
            s = s[end+2:].lstrip()

        # 去除单行注释 --
        while s.startswith("--"):
            newline = s.find("\n")
            if newline == -1:
                return ""
            s = s[newline+1:].lstrip()

        # 最终提取首个 token
        first_token = s.split()[0].upper() if s else ""
        return first_token

    def _check_sql_safe(self, sql: str):
        """
        确定 SQL 是否安全可执行：
        1. 必须是 SELECT / SHOW / DESCRIBE / EXPLAIN 开头
        2. 禁止出现各种修改类语句
        """
        prefix = self._clean_sql_prefix(sql)

        if not prefix:
            raise Exception("无法解析 SQL，请输入正确的查询语句。")

        # 如果不是 select/show/describe/explain 开头，直接禁止
        if prefix not in self.ALLOWED_SQL_PREFIXES:
            raise Exception(f"禁止执行非查询语句：'{prefix}'。仅允许 SELECT/SHOW/DESCRIBE/EXPLAIN。")

        # 进一步检查是否包含危险关键字（防止在子查询中藏修改）
        upper_sql = sql.upper()
        for kw in self.FORBIDDEN_SQL_KEYWORDS:
            if kw in upper_sql:
                raise Exception(f"检测到危险 SQL 关键字 '{kw}'，禁止执行修改或危险操作。")

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        sql_params = {
            "host": tool_parameters.get("host"),
            "port": tool_parameters.get("port"),
            "user": tool_parameters.get("user"),
            "password": tool_parameters.get("password"),
            "database": tool_parameters.get("database")
        }
        execute_sql = tool_parameters.get("execute_sql")

        if not all(sql_params.values()):
            raise Exception("数据库连接参数不能为空")

        if not execute_sql or not isinstance(execute_sql, str):
            raise Exception("execute_sql 必须是非空字符串")

        # ★ 核心防护：严格校验 SQL 类型 ★
        self._check_sql_safe(execute_sql)

        conn = pymysql.connect(**sql_params, charset="utf8mb4", cursorclass=DictCursor)
        cursor = conn.cursor()

        try:
            cursor.execute(execute_sql)
            rows = cursor.fetchall()
            yield self.create_json_message({"result": rows})

        except pymysql.err.ProgrammingError as e:
            error_code, error_msg = e.args

            # 针对表不存在等逻辑错误，让 LLM 自己改
            if error_code == 1146:
                raise Exception({
                    "status": "error",
                    "message": f"执行失败: {error_msg}。请不要猜测表名，可先执行 'SHOW TABLES' 查看现有表。"
                })
            else:
                raise Exception({"status": "error", "message": f"SQL执行错误: {error_msg}"})

        except Exception as e:
            raise Exception(f"调用失败: {e}")

        finally:
            cursor.close()
            conn.close()
