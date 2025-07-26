#!/usr/bin/env python3
"""
全局日志管理器
统一管理所有模块的日志输出，优先发送到Live面板
"""

import logging
from rich.console import Console
from rich.logging import RichHandler
from typing import Optional
import threading

# 配置传统日志作为后备
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
_backup_logger = logging.getLogger("GlobalLogger")

class GlobalLogManager:
    """全局日志管理器，单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._live_display = None
        self._lock = threading.Lock()
        self._initialized = True
    
    def set_live_display(self, live_display):
        """设置Live面板显示器"""
        with self._lock:
            self._live_display = live_display
    
    def log_message(self, text: str, level: str = "info", module_name: str = ""):
        """
        发送日志消息
        
        Args:
            text: 日志文本
            level: 日志级别 (info, warning, error, success)
            module_name: 模块名称（可选，用于标识消息来源）
        """
        # 如果提供了模块名称，添加到消息前缀
        if module_name:
            formatted_text = f"[{module_name}] {text}"
        else:
            formatted_text = text
        
        with self._lock:
            if self._live_display:
                # 优先发送到Live面板
                self._live_display.add_message(formatted_text, level)
            else:
                # 如果Live面板还未初始化，使用传统logger作为后备
                if level == "info":
                    _backup_logger.info(formatted_text)
                elif level == "warning":
                    _backup_logger.warning(formatted_text)
                elif level == "error":
                    _backup_logger.error(formatted_text)
                elif level == "success":
                    _backup_logger.info(formatted_text)

# 全局日志管理器实例
_global_log_manager = GlobalLogManager()

def set_live_display(live_display):
    """设置Live面板显示器（全局函数接口）"""
    _global_log_manager.set_live_display(live_display)

def log_message(text: str, level: str = "info", module_name: str = ""):
    """发送日志消息（全局函数接口）"""
    _global_log_manager.log_message(text, level, module_name)

# 便捷函数
def log_info(text: str, module_name: str = ""):
    """发送info级别日志"""
    log_message(text, "info", module_name)

def log_warning(text: str, module_name: str = ""):
    """发送warning级别日志"""
    log_message(text, "warning", module_name)

def log_error(text: str, module_name: str = ""):
    """发送error级别日志"""
    log_message(text, "error", module_name)

def log_success(text: str, module_name: str = ""):
    """发送success级别日志"""
    log_message(text, "success", module_name)