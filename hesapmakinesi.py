#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:52:47 2025

@author: velibilir

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:05:30 2025

@author: velibilir
"""
#İşlemlerin sonucunu tutmak için başlangıç değeri
total = None
class operations:#işlemlerin yapılacağı sınıf

    #Toplama işlemi
    @staticmethod
    def add(total,secondNumber):
        return total + secondNumber

    #Çıkarma işlemi
    @staticmethod
    def subtract(total,secondNumber):
        return  total - secondNumber
    
    #Çarpma işlemi
    @staticmethod
    def multiply(total,secondNumber):
        return total*secondNumber
    
    #Bölme işlemi,
    @staticmethod
    def divide(total,secondNumber):
        if secondNumber == 0:
            return "Hata: Sıfıra bölme tanımsızdır!"
        return  total/secondNumber
    
    #Yüzde hesaplama işlemi
    @staticmethod
    def square_root(total,secondNumber):
        return (total/100)*secondNumber
       
    #Karakök alma işlemi
    @staticmethod
    def root(total):
        if total < 0:
            return "Hata: Negatif sayıların karekökü alınamaz."
        return total**0.5 
    
    #Karakök alma işlemi n1'in n2'nci kökü
    @staticmethod
    def percentage(total,secondNumber):
        if secondNumber<= 0:
            return "Hata: Geçersiz kök derecesi!"
        return total**(1/secondNumber)

    #Sonsuz döngü ile kullanıcıdan giriş alınır
while True:
    if total is None:
        #kulanıcıdan ilk sayı istenir
        entry = input("Bir sayı girin (Çıkış için 'esc' yazın): ")
        
        if entry.lower() == 'esc':  #Çıkış komutu
            break

        try:
            total= float(entry)
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
            continue
     #Kullanıcıya grandTotal sonucu göstermek ve hafızayı sıfırlama işlemleri sunar
    entry=input(" Grand Totali görmek için '=' basın veya hafızayı sıfırlamak için 'reset' yazın yoksa 'enter' bas:")
    if entry == '=':
         print("Grand Total:", total)
         continue
     
    if entry.lower() == "reset":
         total = None
         print("Hafıza sıfırlandı!")
         continue
     #Kullanıcıya mevcut işelemleri sunmak
    print("""
Toplama   : 1
Çıkarma   : 2
Çarpma    : 3
Bölme     : 4
Kökalma   : 5
Karakök   : 6
Yüzde Alma: 7
""")

    
    try:
        operation = int(input("Yapmak istediğiniz işlemin numarasını yazınız: "))
    except ValueError:
        print("Lütfen geçerli bir işlem numarası girin.")
        continue
    secondNumber=None
    # İkinci sayı gerektiren işlemler için kullanıcıdan sayı girişi alınır
    if operation in [1, 2, 3, 4, 5, 7]:  
        entry = input("İkinci sayıyı girin: ")
        
        try:
            secondNumber = float(entry)
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
            continue
        #İşlem sınıf örneği çağrılır
    
    
    #işleme göre hesaplamalar yapılır
    if operation == 1:
        total = operations.add(total,secondNumber)
    elif operation == 2:
        total = operations.subtract(total,secondNumber)
    elif operation == 3:
        total = operations.multiply(total,secondNumber)
    elif operation == 4:
        total = operations.divide(total,secondNumber)
    elif operation == 5:
        total = operations.square_root(total,secondNumber)
    elif operation == 6:
        total = operations.root(total)
    elif operation == 7:
        total = operations.percentage(total,secondNumber)
    else:
        print("Geçerli bir işlem giriniz.")
        continue
    #ekrana bastırılır
    print("Sonuç:", total)

