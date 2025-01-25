#include <WiFi.h>
#include "esp_wifi.h"

#define WIFI_SSID "YOUR_WIFI_SSID"       // Wi-Fi SSID
#define WIFI_PASS "YOUR_WIFI_PASSWORD"   // Wi-Fi Password

// CSI 데이터 수신 콜백 함수
void csi_callback(void *ctx, wifi_csi_info_t *csi_info) {
    Serial.println("CSI Data Received:");
    Serial.print("CSI Length: ");
    Serial.println(csi_info->len);

    Serial.print("CSI Data: ");
    for (int i = 0; i < csi_info->len; i++) {
        Serial.print(csi_info->buf[i]);
        Serial.print(" ");
    }
    Serial.println();
}

void setup() {
    Serial.begin(115200);

    // Wi-Fi 초기화 및 연결
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.println("Connecting to Wi-Fi...");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWi-Fi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // CSI 설정
    wifi_csi_config_t csi_config = {
        .lltf_en = true,  // 장기 파일럿 CSI 활성화
        .htltf_en = true, // 고속 파일럿 CSI 활성화
        .stbc_en = false, // STBC CSI 비활성화
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
    };

    // CSI 콜백 등록 및 설정
    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(csi_callback, NULL);

    Serial.println("CSI measurement started.");
}

void loop() {
    // Do nothing, CSI 데이터는 콜백에서 처리됨.
}