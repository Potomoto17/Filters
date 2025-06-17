import cv2 as cv
import numpy as np

def konvolucija(slika, jedro):
    # Slika
    # 1  2  3
    # 4  5  6
    # 7  8  9
    # Jedro
    # -1  0  1
    # -2  0  2
    # -1  0  1
    # Izračun
    # (-1×1) + (0×2) + (1×3) +
    # (-2×4) + (0×5) + (2×6) +
    # (-1×7) + (0×8) + (1×9) = 0
    # To je izračun za določen del po katerem se jedro slajda

    # Določimo potrebne parametre
    višina, širina = slika.shape
    k_višina, k_širina = jedro.shape
    
    # Določimo velikost obrobe (padding) glede na velikost jedra
    pad_h = k_višina // 2 # Height
    pad_w = k_širina // 2 # Width
    
    # Razširimo sliko z ničlami na robovih
    # Pad je funkcija, ki dodaja obrobe, reflect je tip obrobe, ki jo bomo uporabljali, baje je boljsi od constant za filtre (source: chat)
    razširjena_slika = np.pad(slika, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant') 
    
    # Inicializiramo rezultatno sliko
    rezultat = np.zeros_like(slika, dtype=np.float32)
    
    # Ročna konvolucija
    for i in range(višina):
        for j in range(širina):
            # Izrežemo malo okno
            izrez = razširjena_slika[i:i+k_višina, j:j+k_širina]
            
            # Izračunamo konvolucijo (glej zgoraj)
            rezultat[i, j] = np.sum(izrez * jedro)
    
    # Ce so stevilke vecje od 255 ali manjse od 0 jih avtomatsko nastavi na 0 ali na 255
    #rezultat = np.clip(rezultat, 0, 255)
    return rezultat

def filtriraj_z_gaussovim_jedrom(slika, sigma):
    # Izračuna velikost jedra po formuli, nato pa s pomočjo te vrednosti zamegli sliko 0'5 = malo, 3'0 = fejst (odvisno od sigme)
    # Velikost jedra po formuli
    velikost_jedra = int((2 * sigma) * 2 + 1)
    k = (velikost_jedra / 2) - 0.5  # Aritmnetična sredina jedra, updated to match the formula k = velikost_jedra/2 - 1/2
    
    # Inicializacija Gaussovega jedra
    jedro = np.zeros((velikost_jedra, velikost_jedra), dtype=np.float32)
    
    # Izračun Gaussovega jedra po formuli za vsak element
    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            x = i - k - 1  # Adjusted to match the formula (i - k - 1)
            y = j - k - 1  # Adjusted to match the formula (j - k - 1)
            jedro[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalizacija jedra (vsota vseh vrednosti mora biti vsaj 1)
    jedro /= np.sum(jedro)
    
    # Konvolucija slike z Gaussovim jedrom
    filtrirana_slika = konvolucija(slika, jedro)
    
    # Convert to uint8 for proper display
    filtrirana_slika = filtrirana_slika.astype(np.uint8)
    
    return filtrirana_slika



def filtriraj_sobel_smer(slika, originalna_slika):
    # Sobelovo jedro za horizontalne robove (po definiciji SOBELx = "spodnja matrika")
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    
    # Konvolucija slike z jedrom (za jedro se uporabi sobelova matrika)
    filtrirana_slika = np.abs(konvolucija(slika, sobel_x))
    
    # Najdemo slikovne elemente z močnim gradientom (150 je meja vse nad 150 bo izbrano)
    prag = 150
    # Vse shranemo v močni_gradienti
    močni_gradienti = filtrirana_slika > prag
    
    # Kopiramo originalno sliko
    barvna_slika = np.copy(originalna_slika)  # Changed to use the original color image
    
    # Označimo močne gradiente z zeleno barvo
    barvna_slika[močni_gradienti] = [0, 255, 0] # Zelena barva
    
    return barvna_slika

if __name__ == '__main__':
    # Preberemo sliko (pot do slike "lenna.png" v mapi .utils)
    slika = cv.imread(".utils/lenna.png")

    # Če slika ni bila uspešno prebrana, izpišemo napako
    if slika is None:
        print("Napaka pri nalaganju slike.")
        exit()

    # Pretvorba slike v sivinsko (če je potrebna za konvolucijo)
    sivinska_slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    # 1. Filtriranje z Gaussovim jedrom
    sigma = 0.5  # RAZLIČNI REZULATIT ČE POVEČAŠ ALI POMANJŠAŠ
    gauss_filtrirana_slika = filtriraj_z_gaussovim_jedrom(sivinska_slika, sigma)
    
    # 2. Filtriranje s Sobelovim jedrom
    sobel_filtrirana_slika = filtriraj_sobel_smer(gauss_filtrirana_slika, slika)  # Pass the original color image

    # Prikaz rezultatov
    cv.imshow("Origibalna Slika Črno Bela", sivinska_slika)
    cv.imshow("Originalna Slika", slika)
    cv.imshow("Filtrirana z Gaussovim Jedrom", gauss_filtrirana_slika)
    cv.imshow("Filtrirana s Sobelovim Jedrom", sobel_filtrirana_slika)
    
    # Čakamo na uporabnikov pritisk tipke
    cv.waitKey(0)
    cv.destroyAllWindows() 

'''
def konvolucija(slika, jedro):
    # Slika
    # 1  2  3
    # 4  5  6
    # 7  8  9
    # Jedro
    # -1  0  1
    # -2  0  2
    # -1  0  1
    # Izračun
    # (-1×1) + (0×2) + (1×3) +
    # (-2×4) + (0×5) + (2×6) +
    # (-1×7) + (0×8) + (1×9) = 0
    # To je izračun za določen del po katerem se jedro slajda

    # Določimo potrebne parametre
    višina, širina = slika.shape
    k_višina, k_širina = jedro.shape
    
    # Določimo velikost obrobe (padding) glede na velikost jedra
    pad_h = k_višina // 2 # Height
    pad_w = k_širina // 2 # Width
    
    # Razširimo sliko z ničlami na robovih
    razširjena_slika = np.pad(slika, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect') 
    
    # Inicializiramo rezultatno sliko
    rezultat = np.zeros_like(slika, dtype=np.float32)
    
    # Ročna konvolucija
    for i in range(višina):
        for j in range(širina):
            izrez = razširjena_slika[i:i+k_višina, j:j+k_širina]
            rezultat[i, j] = np.sum(izrez * jedro)
    
    rezultat = np.clip(rezultat, 0, 255)
    return rezultat

def filtriraj_z_gaussovim_jedrom(slika, sigma):
    velikost_jedra = int((2 * sigma) * 2 + 1)
    k = (velikost_jedra / 2) - 0.5  
    
    jedro = np.zeros((velikost_jedra, velikost_jedra), dtype=np.float32)
    
    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            x = i - k - 1  
            y = j - k - 1  
            jedro[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    jedro /= np.sum(jedro)
    
    filtrirana_slika = konvolucija(slika, jedro)
    filtrirana_slika = filtrirana_slika.astype(np.uint8)
    
    return filtrirana_slika

def filtriraj_sobel_smer(slika, originalna_slika):
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    
    filtrirana_slika = konvolucija(slika, sobel_x)
    
    prag = 150
    močni_gradienti = filtrirana_slika > prag
    
    barvna_slika = np.copy(originalna_slika)  # Changed to use the original color image
    barvna_slika[močni_gradienti] = [0, 255, 0] # Zelena barva
    
    return barvna_slika

# BELO DODANO: Nova funkcija za obdelavo sivinskih slik s Sobelovim filtrom
def filtriraj_sobel_smer_sivinska(slika):
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    
    filtrirana_slika = konvolucija(slika, sobel_x)
    filtrirana_slika = np.clip(filtrirana_slika, 0, 255)
    filtrirana_slika = filtrirana_slika.astype(np.uint8)
    
    return filtrirana_slika
# KONEC BELE DODANE FUNKCIJE

if __name__ == '__main__':
    slika = cv.imread(".utils/lenna.png")

    if slika is None:
        print("Napaka pri nalaganju slike.")
        exit()

    sivinska_slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    sigma = 0.5
    gauss_filtrirana_slika = filtriraj_z_gaussovim_jedrom(sivinska_slika, sigma)
    
    sobel_filtrirana_slika = filtriraj_sobel_smer(gauss_filtrirana_slika, slika)  # barvna
    
    # BELO DODANO: Uporabi Sobel filter tudi za sivinsko sliko
    sobel_sivinska = filtriraj_sobel_smer_sivinska(gauss_filtrirana_slika)
    # KONEC BELE DODANE VRSTICE

    # Prikaz rezultatov
    cv.imshow("Originalna Slika", gauss_filtrirana_slika)
    cv.imshow("Filtrirana s Sobelovim Jedrom (barvna)", sobel_filtrirana_slika)

    # BELO DODANO: Prikaz črno-belega Sobela
    cv.imshow("Filtrirana s Sobelovim Jedrom (sivinska)", sobel_sivinska)
    # KONEC BELE DODANE VRSTICE

    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
