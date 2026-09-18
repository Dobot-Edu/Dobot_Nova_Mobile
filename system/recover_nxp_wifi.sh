#!/bin/bash
# NXP 88W9098 固件异常后，mlan0 会完全消失。
# 本脚本只在接口消失时恢复驱动，不处理普通掉线，也不修改路由或网络共享。

set -u

WIFI_INTERFACE="mlan0"
WPA_SERVICE="rb-jetson-service-net-wifi-wpa_supplicant_mlan0.service"

if [[ -e "/sys/class/net/${WIFI_INTERFACE}" ]]; then
    exit 0
fi

logger -t nxp-wifi-recover "${WIFI_INTERFACE} disappeared; reloading NXP Wi-Fi driver"

systemctl stop "${WPA_SERVICE}" || true

# 固件失去响应时卸载可能较慢，限制每一步等待时间。
timeout --signal=INT 30s modprobe -r moal || true
timeout --signal=INT 10s modprobe -r mlan || true
sleep 2

modprobe mlan
modprobe moal mod_para=nxp/wifi_mod_para.conf
udevadm settle --timeout=10 || true

if [[ ! -e "/sys/class/net/${WIFI_INTERFACE}" ]]; then
    logger -t nxp-wifi-recover "recovery failed: ${WIFI_INTERFACE} is still missing"
    exit 1
fi

systemctl restart "${WPA_SERVICE}"
logger -t nxp-wifi-recover "${WIFI_INTERFACE} recovered"
