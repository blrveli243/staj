#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:05:30 2025

@author: velibilir
"""

sonuc = None
class islemler:
    def __init__(self,sonuc, n2=None):
        self.n1 = sonuc
        self.n2 = n2
    
    def toplama(self):
        return self.n1 + self.n2
    
    def cikarma(self):
        return self.n1 - self.n2
    
    def carpma(self):
        return self.n1 * self.n2
    
    def bolme(self):
        if self.n2 == 0:
            return "Hata: Sıfıra bölme tanımsızdır!"
        return self.n1 / self.n2
    
    def yuzdealma(self):
        return (self.n1 / 100) * self.n2
    
    def karekok(self):
        if self.n1 < 0:
            return "Hata: Negatif sayıların karekökü alınamaz."
        return self.n1 ** 0.5
    
    def kokalma(self):
        if self.n2 <= 0:
            return "Hata: Geçersiz kök derecesi!"
        return self.n1 ** (1 / self.n2)

    
while True:
    if sonuc is None:
        giris = input("Bir sayı girin (Çıkış için 'esc' yazın): ")
        
        if giris.lower() == 'esc':  
            break

        try:
            sonuc = float(giris)
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
            continue
     
    giris=input(" Grand Totali görmek için '=' basın veya hafızayı sıfırlamak için 'reset' yazın yoksa 'enter' bas:")
    if giris == '=':
         print("Grand Total:", sonuc)
         continue
     
    if giris.lower() == "reset":
         sonuc = None
         print("Hafıza sıfırlandı!")
         continue
     
    print("""
Toplama   : 1
Çıkarma   : 2
Çarpma    : 3
Bölme     : 4
Karekök   : 5
Kökalma   : 6
Yüzde Alma: 7
""")

    
    try:
        islem = int(input("Yapmak istediğiniz işlemin numarasını yazınız: "))
    except ValueError:
        print("Lütfen geçerli bir işlem numarası girin.")
        continue
    n2=None
    if islem in [1, 2, 3, 4, 6, 7]:  
        giris = input("İkinci sayıyı girin: ")
        
        try:
            n2 = float(giris)
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
            continue
        
    calc = islemler(sonuc, n2)
    
    
    if islem == 1:
        sonuc = calc.toplama()
    elif islem == 2:
        sonuc = calc.cikarma()
    elif islem == 3:
        sonuc = calc.carpma()
    elif islem == 4:
        sonuc = calc.bolme()
    elif islem == 5:
        sonuc = calc.karekok()
    elif islem == 6:
        sonuc = calc.kokalma()
    elif islem == 7:
        sonuc = calc.yuzdealma()
    else:
        print("Geçerli bir işlem giriniz.")
        continue
    
    print("Sonuç:", sonuc)
