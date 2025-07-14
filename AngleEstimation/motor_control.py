#!/usr/bin/env python3
"""
Prosty skrypt testowy do sterowania silnikiem krokowym
"""

import socket
import time

def send_step_command(esp_ip, command):
    """Wyślij komendę do ESP32 i odbierz odpowiedź"""
    try:
        # Utwórz połączenie
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((esp_ip, 3333))
        
        # Wyślij komendę
        sock.send(command.encode())
        print(f"Wysłano: {command}")
        
        # Odbierz odpowiedź
        response = sock.recv(1024).decode().strip()
        print(f"Odpowiedź: {response}")
        
        sock.close()
        return response
        
    except Exception as e:
        print(f"Błąd: {e}")
        return None

def main():
    # ZMIEŃ NA IP TWOJEGO ESP32
    ESP32_IP = "192.168.0.105"  # ⚠️ WPROWADŹ RZECZYWISTY IP
    
    print("=== Test silnika krokowego ===")
    print(f"ESP32 IP: {ESP32_IP}")
    
    # Test połączenia
    print("\n1. Test połączenia...")
    response = send_step_command(ESP32_IP, "STATUS")
    if response != "READY":
        print("❌ ESP32 nie odpowiada poprawnie!")
        return
    
    print("✅ Połączenie OK")
    
    # Test kroków
    print("\n2. Test 5 kroków w prawo...")
    for i in range(5):
        response = send_step_command(ESP32_IP, "STEP_CW")
        if response == "DONE_CW":
            print(f"✅ Krok {i+1} wykonany")
        else:
            print(f"❌ Błąd kroku {i+1}")
        time.sleep(0.5)
    
    print("\n3. Pauza 2 sekundy...")
    time.sleep(2)
    
    print("\n4. Test 5 kroków w lewo...")
    for i in range(5):
        response = send_step_command(ESP32_IP, "STEP_CCW")
        if response == "DONE_CCW":
            print(f"✅ Krok {i+1} wykonany")
        else:
            print(f"❌ Błąd kroku {i+1}")
        time.sleep(0.5)
    
    print("\n✅ Test zakończony!")

if __name__ == "__main__":
    main()