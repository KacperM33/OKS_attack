# *Ataki adwersarialne na wybrane architektury sieci neuronowych w zadaniach analizy wideo*

> 🇬🇧 **English version / ACIVS Paper:**  
> Switch to the main paper repository README: [README.md](README.md).

Zaproponowany **OKS Attack** jest czarnoskrzynkowym atakiem adwersarialnym wykorzystującym metrykę **OKS**, określającą stopień podobieństwa anatomicznych punktów kluczowych, jako informację zwrotną ataku. Weryfikację podatności modeli przeprowadzono z wykorzystaniem zbioru Penn Action, zawierającego sekwencje wideo z zarejestrowanymi czynnościami sportowymi. Przeprowadzone eksperymenty wykazały, że badane architektury wykazują podatność na generowane perturbacje w obydwu wiodących paradygmatach estymacji pozy: oddolnym (jednoetapowym) oraz odgórnym (dwuetapowym).

## ℹ️ INFO

Kod źródłowy oraz instrukcja instalacji są wspólne dla obu wariantów badań i zostały szczegółowo opisane w głównym pliku [README.md](README.md).

Główną różnicą jest to, że w ramach pracy magisterskiej przeprowadzono **alternatywne warianty eksperymentów**. Badają one działanie perturbacji optymalizowanych w przeciwnym kierunku niż miało to miejsce w przypadku artykułu naukowego.

Instrukcja odtworzenia tych konkretnych eksperymentów została przedstawiona poniżej w punkcie [REPRODUKCJA EKSPERYMENTÓW](#reproducing-experiments).


## SPIS TREŚCI

🔹 [📊 WYNIKI](#main-results)
<br>🔹 [🧪 REPRODUKCJA EKSPERYMENTÓW](#reproducing-experiments) 

<a id="main-results"></a>
## 📊 WYNIKI

**Rysunek 1.** Schemat przeprowadzanego ataku.

<img width="600" alt="fig3_5" src="https://github.com/user-attachments/assets/4834a479-2c8d-4622-b5c5-d0e20714140d" />

**Rysunek 2.** Reprezentacja błędnych klasyfikacji akcji. Każdy wiersz pokazuje klatki z jednego wideo w odstępie ∆ = 10 klatek. Oryginalne oraz adwersarialne predykcje póz zostały odpowiednio zaprezentowane zielonym oraz czerwonym kolorem; brak szkieletu odpowiada za brak detekcji. Prostokąty z prawej pokazują pewność predykcji w danym wideo.

<img width="800" alt="fig2_2" src="https://github.com/user-attachments/assets/26c957fd-2c00-4c1b-98c7-2789abd55a26" />

**Tabela 1.** Średni OKS dla czystych i adwersarialnych póz wyznaczonych przez estymator pozy. Wyniki uwzględnione zostały dla wszystkich wideo, wideo ze zmienioną akcją oraz wideo bez zmienionej akcji.

<center>
<table>
  <thead>
    <tr>
      <th rowspan="3">Model</th>
      <th colspan="6"><center>Średni OKS</center></th>
    </tr>
    <tr>
      <th colspan="2"><center>Wszystkie wideo</center></th>
      <th colspan="2"><center>Akcja zmieniona</center></th>
      <th colspan="2"><center>Akcja niezmieniona</center></th>
    </tr>
    <tr><center>
      <th><center>Orig</th>
      <th><center>OKS Atak</th>
      <th><center>Orig</th>
      <th><center>OKS Atak</th>
      <th><center>Orig</th>
      <th><center>OKS Atak</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet-50</td>
      <td><center>0.7298</td>
      <td><b><center>0.6132</b></td>
      <td><center>0.6548</td>
      <td><b><center>0.4826</b></td>
      <td><center>0.7505</td>
      <td><b><center>0.6492</b></td>
    </tr>
    <tr>
      <td>MobileNetV2</td>
      <td><center>0.6952</td>
      <td><b><center>0.5556</b></td>
      <td><center>0.5631</td>
      <td><b><center>0.3744</b></td>
      <td><center>0.7354</td>
      <td><b><center>0.6106</b></td>
    </tr>
    <tr>
      <td>YOLO-Pose M</td>
      <td><center>0.7393</td>
      <td><b><center>0.6668</b></td>
      <td><center>0.6121</td>
      <td><b><center>0.5048</b></td>
      <td><center>0.7854</td>
      <td><b><center>0.7255</b></td>
    </tr>
    <tr>
      <td>YOLO-Pose S</td>
      <td><center>0.7149</td>
      <td><b><center>0.6024</b></td>
      <td><center>0.6144</td>
      <td><b><center>0.4476</b></td>
      <td><center>0.7676</td>
      <td><b><center>0.6836</b></td>
    </tr>
  </tbody>
</table>
</center>

<a id="reproducing-experiments"></a>
## 🧪 REPRODUKCJA EKSPERYMENTÓW

### 1. Generowanie podzbioru

W celu wygenerowania niestandardowego zbioru dla eksperymentów należy użyć skryptu `select_videos.py`.

> **Reprodukcja wyników pracy magisterskiej:** Aby odtworzyć dokładne wyniki uzyskane w pracy magisterskiej, **możesz pominąć ten krok**. Do badań użyto 20% zbioru danych podzielonego na 20 batchy. Gotowe pliki `.txt` z podziałem są umieszone w repozytorium w katalogu `penn-action-dataset/subset_selected_files_20p_20b/`.

Przykładowa komenda do wygenerowania niestandardowego podzbioru (najpierw przejść do katalogu `oks_tools/`):

```bash
python select_videos.py --labels ../penn-action-dataset/labels --output ../penn-action-dataset/subset_selected_files_20p_20b --batches 20 --percent 20
```
Argumenty:

- `--labels` - Katalog zawierający pliki .mat z anotacjami ze zbioru Penn Action. 

- `--output` - Katalog, w którym zostaną zapisane wygenerowane pliki .txt (zawierające nazwy plików wideo).

- `--batches` - Liczba batchy, na ile zostanie podzielony podzbiór (domyślnie: 1).

- `--percent` - Procent (%) całego zbioru do wydzielenia (domyślnie: 10).

### 2. Zmiany w kodzie

Domyślnie repozytorium zawiera konfigurację do reprodukcji wyników zaprezentowanych w artykule **ACIVS 2026**. Aby odtworzyć wyniki z realizowanej pracy magisterskiej, należy:

1. Otworzyć plik z atakiem z folderu `attacks/`:
    
   - `attacks/attack_core.py` dla wartiantów dwuetapowych
   - `attacks/OKS_attack_oneshot.py` dla wariantów jednoetapowych
2. Znaleźć fragmenty dotyczące perturbacji.
3. Zakomentować domyślne linie i odkomentować linie zaznaczone jako:

    `[Master's Thesis Variant]`

### 3. Uruchomienie eksperymentu

Do ewaluacji modeli i uruchomienia ataku adwersarialnego na wybranych wideo, należy użyć jednego ze skryptów eksperymentalnych.

Przykładowa komenda (najpierw przejść do katalogu `oks_experiments/`):
```bash
python det_oracle_bbox_experiment.py --input ../penn-action-dataset/subset_selected_files_20p_20b/selected_files_batch_0.txt --output experiments_test --model res50
```
Argumenty:

- `--input` - Plik .txt zawierający nazwy plików wideo (pozyskany z `select_videos.py` lub z dostępnego podziału).

- `--output` - Katalog, w którym zostaną zapisane logi i wyniki eksperymentu.

- `--model` - Model estymacji pozy, który zostanie użyty w przeprowadzanym eksperymecie.

### 4. Skrypty eksperymentalne

 | Skrypt |  Typ estymatora | Atak | Badanie ablacyjne |
 | :--- | :--- | :--- | :--- |
 | `det_oracle_bbox_experiment.py` | dwuetapowy | `attacks/OKS_attack_det.py` | - |
 | `oneshot_experiment.py` | jednoetapowy | `attacks/OKS_attack_oneshot.py` | - |
 | `no_det_oracle_bbox_experiment.py` | dwuetapowy | `attacks/OKS_attack_no_det.py` | ☑️ | 
 | `no_det_clean_bbox_experiment.py` | dwuetapowy | `attacks/OKS_attack_no_det.py` | ☑️ | 
