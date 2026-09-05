#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口检测测试工具
用于检查设备中可用的串口设备并显示详细信息
"""

import serial
import serial.tools.list_ports
import platform
import sys
from typing import List, Dict, Any


def get_available_serial_ports() -> List[Dict[str, Any]]:
    """
    获取所有可用的串口设备
    
    Returns:
        List[Dict[str, Any]]: 串口设备信息列表
    """
    ports = []
    available_ports = serial.tools.list_ports.comports()
    
    for port in available_ports:
        port_info = {
            'device': port.device,
            'name': port.name,
            'description': port.description,
            'hwid': port.hwid,
            'vid': port.vid,
            'pid': port.pid,
            'serial_number': port.serial_number,
            'manufacturer': port.manufacturer,
            'product': port.product,
            'interface': port.interface
        }
        ports.append(port_info)
    
    return ports


def test_serial_port_connection(port_name: str, baudrate: int = 9600, timeout: float = 1.0) -> bool:
    """
    测试串口是否可以正常连接
    
    Args:
        port_name (str): 串口设备名
        baudrate (int): 波特率，默认9600
        timeout (float): 超时时间，默认1秒
        
    Returns:
        bool: 连接成功返回True，否则返回False
    """
    try:
        with serial.Serial(port_name, baudrate, timeout=timeout) as ser:
            return True
    except Exception as e:
        print(f"    连接测试失败: {e}")
        return False


def print_system_info():
    """打印系统信息"""
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"PySerial版本: {serial.__version__}")
    print("-" * 60)


def print_port_details(port_info: Dict[str, Any], index: int):
    """
    打印串口详细信息
    
    Args:
        port_info (Dict[str, Any]): 串口信息字典
        index (int): 串口编号
    """
    print(f"\n串口 #{index + 1}:")
    print(f"  设备路径: {port_info['device']}")
    print(f"  设备名称: {port_info['name']}")
    print(f"  描述信息: {port_info['description']}")
    
    if port_info['manufacturer']:
        print(f"  制造商: {port_info['manufacturer']}")
    
    if port_info['product']:
        print(f"  产品名: {port_info['product']}")
    
    if port_info['serial_number']:
        print(f"  序列号: {port_info['serial_number']}")
    
    if port_info['vid'] is not None:
        print(f"  厂商ID (VID): 0x{port_info['vid']:04X}")
    
    if port_info['pid'] is not None:
        print(f"  产品ID (PID): 0x{port_info['pid']:04X}")
    
    print(f"  硬件ID: {port_info['hwid']}")
    
    # 测试连接
    print(f"  连接测试: ", end="")
    if test_serial_port_connection(port_info['device']):
        print("✓ 成功")
    else:
        print("✗ 失败")


def main():
    """主函数 - 检测并显示所有可用串口"""
    print("=" * 60)
    print("             串口设备检测工具")
    print("=" * 60)
    
    # 打印系统信息
    print_system_info()
    
    # 获取可用串口
    ports = get_available_serial_ports()
    
    if not ports:
        print("❌ 未检测到可用的串口设备")
        print("\n可能的原因:")
        print("  1. 没有连接任何串口设备")
        print("  2. 设备驱动程序未正确安装")
        print("  3. 权限不足（Linux/Mac下可能需要sudo权限）")
        return
    
    print(f"✓ 检测到 {len(ports)} 个可用串口设备:\n")
    
    # 显示每个串口的详细信息
    for i, port in enumerate(ports):
        print_port_details(port, i)
    
    print("\n" + "=" * 60)
    print("检测完成!")
    
    # 提供使用建议
    print("\n使用建议:")
    print("  - 选择连接测试成功的串口进行通信")
    print("  - 确认设备的波特率、数据位、停止位等参数")
    print("  - 在Linux系统下，可能需要将用户添加到dialout组:")
    print("    sudo usermod -a -G dialout $USER")


if __name__ == "__main__":
    main()
