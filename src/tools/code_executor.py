from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any
from src.tools.base import BaseTool

@dataclass
class CodeExecutorTool(BaseTool):
    name:str = "code_executor"
    description: str = "Execute Python code in a sandboxed subprocess"

    def _default_parameters(self) -> dict[str,Any]:
        return {
            "type":"object",
            "properties":{
                "code":{
                    "type":"string",
                    "description":"python code to execute",
                },
                "timeout":{
                    "type":"integer",
                    "description":"timeout in seconds (default 10)",
                },
            },
            "required":["code"],
        }
    async def run(self,code:str = "",timeout:int = 10,**kwargs:Any) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",       # 写入模式（文本）
            suffix=".py",   # 文件扩展名，让 Python 识别
            delete=False,   # 离开 with 块时不自动删除（后面手动删除）
            encoding="utf-8"
        ) as f:
            f.write(code)   # 将代码写入临时文件
            tmp_path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",   # 要执行的程序：Python 解释器
                tmp_path,
                stdout=subprocess.PIPE,     # 捕获标准输出 将子进程的输出管道化，供父进程读取
                stderr=subprocess.PIPE,     # 捕获标准错误 将子进程的错误输出管道化
            )
            stdout,stderr = await asyncio.wait_for(
                proc.communicate(),     # 等待子进程结束，返回 (stdout, stderr) 字节数据
                timeout=timeout,
            )
            out = stdout.decode("utf-8",errors="replace").strip()
            err = stderr.decode("utf-8",errors="replace").strip()
            # .decode("utf-8")将字节转换为字符串,errors="replace"：遇到无法解码的字符用 � 替换，不崩溃

            result = out
            if err:
                result += f"\n\n[stderr]:\n{err}"
            return result or "(no output)"
        except asyncio.TimeoutError:
            return f"[Timeout after {timeout}s]"    #返回超时提示，不删除临时文件（finally 会执行）
        finally:
            os.unlink(tmp_path)     # 删除临时文件

