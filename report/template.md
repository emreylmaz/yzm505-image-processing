# YZM505 Vize Ödevi — Rapor

**Ders:** YZM505 Görüntü İşleme · **Öğrenci:** Emre Yılmaz

Bu rapordaki tüm sayılar `outputs/qN/metrics.json` dosyalarından, tüm kod blokları ise `src/ip/` altındaki gerçek kaynak koddan otomatik olarak alınmıştır (`tools/build_report.py`). Tüm figürler script'ler tarafından üretilmiştir; elle alınmış ekran görüntüsü yoktur.

## 0. Ortam, Yeniden Üretilebilirlik ve Varsayımlar

**Ortam:** Python 3.13, NumPy, OpenCV, scikit-image, matplotlib, tifffile (bağımlılıklar `uv.lock` ile sabit).

```bash
uv run pytest                    # birim testleri (manual vs built-in, el hesabı kontrolleri)
uv run python scripts/q1.py      # q2.py, q3.py → outputs/qN/
uv run python tools/verify_outputs.py
bash report/build.sh             # bu rapor (.md + .docx)
```

**Genel kurallar.** "Manual" fonksiyonlar çekirdek algoritma için OpenCV/scikit-image çağırmaz; yalnızca NumPy dizi işlemleri kullanır. Built-in fonksiyonlar yalnızca `*_builtin` adlı karşılaştırma sarmalayıcılarında kullanılır. Tüm hesaplar `float64` ile yapılır; [0, 255] aralığına clip ve `uint8` dönüşümü yalnızca en sonda uygulanır (`np.rint`: .5 değerleri çift sayıya yuvarlanır, OpenCV `saturate_cast` ile aynı davranış).

**Doğrulama yaklaşımı.** Her manual fonksiyon için (i) küçük sentetik dizilerde el ile türetilmiş değerlerle, (ii) aynı border modu kullanılarak OpenCV karşılığıyla karşılaştıran otomatik testler yazılmıştır. Ayrıca kodu yazan süreçten bağımsız bir doğrulayıcı, beklenen değerleri yalnızca matematiksel tanımlardan türetip fonksiyonlarla karşılaştırmıştır (`tests/test_verifier_q*.py`).

### Varsayımlar (ASSUMPTION)

1. **Q1 — "veya":** Sınav "Contrast Stretching **veya** Histogram Equalization" istemektedir; bu çalışmada **ikisi de** manuel olarak uygulanmış ve karşılaştırılmıştır.
2. **Q1 — gürültü:** "İyileştirme sonrası ortaya çıkan gürültü" ifadesi, kontrast artırmanın görüntüdeki mevcut düşük genlikli gürültüyü büyütmesi olarak yorumlanmıştır; ana analiz bu doğal gürültü üzerindedir. Mean ve Median arasındaki farkı ölçülebilir kılmak için **ek bir deney** olarak HE çıktısına sabit tohumlu (seed = {{q1.pout.salt_pepper.seed}}) salt & pepper gürültüsü (p = {{q1.pout.salt_pepper.p}}) eklenmiştir.
3. **Q1 — filtre girişi:** Filtreler hem Contrast Stretching hem HE çıktısına uygulanmıştır; ana görsel anlatım HE üzerindendir (gürültüyü en çok HE büyütür), her iki koldaki tüm sayılar tablolarda verilmiştir.
4. **Q1 — filtre parametreleri:** Mean ve Median için 3×3 ve 5×5; Gaussian için σ = 1 (7×7) ve σ = 2 (13×13), kernel boyutu 2⌈3σ⌉+1.
5. **Q2 — parametreler:** Shear katsayısı sınavda verilmemiştir; yatay shear k = {{q2.shear_k}} seçilmiştir. Canny eşikleri ({{q2.canny_thresholds.0}}, {{q2.canny_thresholds.1}}) tüm görüntülerde sabittir. Rotation, kırpma olmaması için tuvali büyüten (expand) şekilde ve görüntü merkezi etrafında yapılmıştır; dışarıda kalan bölge 0 (siyah) ile doldurulur.
6. **Q2 — operatörler:** Sobel ve Canny bu soruda dönüşümleri analiz eden araçlardır; OpenCV ile hesaplanmıştır. Geometrik dönüşümler ve her iki interpolasyon manueldir.
7. **Q3 — boyut:** Çıkış boyutu round(1.7·N) olarak alınmıştır ({{q3.input.shape}} → {{q3.resize.out_shape}}). Koordinat oranı, OpenCV'nin `dsize` verildiğinde kullandığı gibi giriş/çıkış boyutu oranıdır (1/1.7 değil).
8. **Giriş formatı:** `.tif` dosyaları `uint8` değilse [0, 255] aralığına lineer ölçeklenir (ör. MATLAB `checkerboard` görüntüsü `double` tipindedir).

---

## 1. Soru 1 — Kontrast İyileştirme ve Gürültü Filtreleme

### 1.1 Yöntem: Contrast Stretching

Lineer germe, görüntünün mevcut [r_min, r_max] aralığını [0, 255] aralığına doğrusal olarak eşler:

$$ s = \frac{r - r_{min}}{r_{max} - r_{min}} \cdot 255 $$

Bu dönüşüm, en koyu pikseli 0'a ve en parlak pikseli 255'e götürdüğü için çıktının tam [0, 255] aralığını kullanmasını garanti eder. Gri seviyeler arasındaki sıralama ve oranlar korunur; histogramın **şekli** değişmez, yalnızca genişler.

{{code:ip.enhance.contrast_stretch}}

### 1.2 Yöntem: Histogram Equalization

Histogram Equalization, gri seviyeleri kümülatif dağılım fonksiyonuna (CDF) göre yeniden eşler. N piksel sayısı, h[k] histogram olmak üzere:

$$ CDF(k) = \sum_{j \le k} h[j], \qquad T(k) = \mathrm{round}\!\left(\frac{CDF(k) - CDF_{min}}{N - CDF_{min}} \cdot 255\right) $$

CDF_min, görüntüdeki en koyu mevcut seviyenin CDF değeridir; bu sayede en koyu seviye 0'a, en parlak seviye 255'e gider. Histogram NumPy `bincount` ile, CDF `cumsum` ile hesaplanır; dönüşüm 256 elemanlı bir tablo (LUT) ile dizi indekslemesi şeklinde uygulanır. Burada yuvarlama `floor(x + 0.5)` (yarımlar yukarı) şeklindedir.

{{code:ip.enhance.equalization_lut}}

{{code:ip.enhance.hist_equalize}}

### 1.3 Sonuçlar: pout.tif

{{fig:q1/pout_fig_enhance.png|pout: orijinal, Contrast Stretching ve Histogram Equalization}}

{{fig:q1/pout_fig_hist.png|pout: histogramlar ve CDF eğrileri}}

{{table:q1_enhance:pout}}

Orijinal pout görüntüsü yalnızca [{{q1.pout.enhance.original.min}}, {{q1.pout.enhance.original.max}}] aralığını kullanmaktadır. Her iki yöntem de çıktıyı tam [0, 255] aralığına yaymıştır. Contrast Stretching histogramın şeklini koruyarak yalnızca genişletir (seviye sayısı {{q1.pout.enhance.stretch.n_levels}}); HE ise CDF'i doğrusallaştırır: CDF'in ideal doğrudan maksimum sapması orijinalde {{q1.pout.enhance.original.cdf_linearity_max_dev|.3f}} iken HE sonrası {{q1.pout.enhance.he.cdf_linearity_max_dev|.3f}} değerine düşmüştür.

**Built-in karşılaştırma:** Manuel HE ile `cv2.equalizeHist` arasındaki maksimum fark {{q1.pout.enhance.he_vs_cv2_equalizeHist.max_abs_diff}} gri seviyedir (aynı formül; olası ±1 fark OpenCV'nin float32 ölçek katsayısı ve yuvarlama kuralından kaynaklanır). `skimage.exposure.equalize_hist` ile fark daha büyüktür (maksimum {{q1.pout.enhance.he_vs_skimage_equalize_hist.max_abs_diff}}, ortalama {{q1.pout.enhance.he_vs_skimage_equalize_hist.mean_abs_diff|.2f}}): scikit-image CDF_min'i çıkarmaz ve [0, 1] aralığında kayan noktalı çıktı üretir; yani formül tanımı farklıdır.

### 1.4 Sonuçlar: tire.tif

{{fig:q1/tire_fig_enhance.png|tire: orijinal, Contrast Stretching ve Histogram Equalization}}

{{fig:q1/tire_fig_hist.png|tire: histogramlar ve CDF eğrileri}}

{{table:q1_enhance:tire}}

Orijinal tire görüntüsü [{{q1.tire.enhance.original.min}}, {{q1.tire.enhance.original.max}}] aralığında ve ortalaması {{q1.tire.enhance.original.mean|.1f}} olan karanlık bir görüntüdür. HE sonrası ortalama {{q1.tire.enhance.he.mean|.1f}} değerine çıkmış, CDF sapması {{q1.tire.enhance.original.cdf_linearity_max_dev|.3f}} → {{q1.tire.enhance.he.cdf_linearity_max_dev|.3f}} olmuştur. Manuel HE ile `cv2.equalizeHist` maksimum farkı: {{q1.tire.enhance.he_vs_cv2_equalizeHist.max_abs_diff}}.

### 1.5 Filtreler: Mean, Median ve Gaussian

Üç filtre de NumPy ile manuel olarak yazılmıştır (pencereler `sliding_window_view` ile oluşturulur). **Border (padding) modları:** Mean ve Gaussian için `reflect101` (…c b | a b c | b a…, OpenCV varsayılanı `BORDER_REFLECT_101`), Median için `replicate` (…a a | a b c | c c…, `cv2.medianBlur`'un sabit kullandığı mod). Aynı border ile OpenCV karşılıklarına göre maksimum fark testlerde ≤ 1 gri seviyedir (OpenCV'nin `uint8` için sabit noktalı aritmetiğinden kaynaklı yuvarlama).

{{code:ip.filters.mean_filter}}

{{code:ip.filters.median_filter}}

{{code:ip.filters.gaussian_kernel}}

{{fig:q1/pout_fig_filters_he.png|pout: HE sonrası tüm filtreler}}

{{fig:q1/tire_fig_filters_he.png|tire: HE sonrası tüm filtreler}}

{{fig:q1/pout_fig_filters_stretch.png|pout: Contrast Stretching sonrası tüm filtreler}}

{{fig:q1/tire_fig_filters_stretch.png|tire: Contrast Stretching sonrası tüm filtreler}}

### 1.6 Seçilen Keskinlik Metriği

> **Benim seçtiğim metrik (edge sharpness):** Nesne kenar maskesi içindeki **ortalama Sobel gradyan büyüklüğü**. Maske, filtrelenmemiş iyileştirilmiş görüntünün Gaussian σ = 2 ile yumuşatılmış kopyasında Canny (50, 150) ile bulunan kenarların 3×3 genişletilmesiyle elde edilir ve **tüm filtreler için aynıdır**. Yumuşatma yalnızca maskeyi bulmak içindir (aksi halde HE'nin büyüttüğü gürültü de "kenar" sayılır); metrik her filtrenin kendi ham çıktısı üzerinde ölçülür.
>
> **Gürültü göstergesi:** Düz bölgelerde (yumuşatılmış görüntüde Sobel büyüklüğü alt %40 ve kenarlardan en az 3 piksel uzak) Laplacian yanıtının standart sapması. Laplacian lineer gölgelemeyi yok saydığı için bu değer yüksek frekanslı gürültüyü ölçer.
>
> **Denge skoru:** Keskinlik korunumu S (%) ve gürültü azalması N (%) için 2·S·N/(S+N) (F1 benzeri harmonik ortalama). İkincil olarak Var(Laplacian) da raporlanmıştır.

**pout — HE sonrası** (kenar maskesi {{q1.pout.filters.he.edge_mask_pixels}} piksel, düz bölge {{q1.pout.filters.he.flat_mask_pixels}} piksel):

{{table:q1_filters:pout:he}}

**tire — HE sonrası** (kenar maskesi {{q1.tire.filters.he.edge_mask_pixels}} piksel, düz bölge {{q1.tire.filters.he.flat_mask_pixels}} piksel):

{{table:q1_filters:tire:he}}

**pout — Contrast Stretching sonrası:**

{{table:q1_filters:pout:stretch}}

**tire — Contrast Stretching sonrası:**

{{table:q1_filters:tire:stretch}}

{{fig:q1/pout_fig_metrics.png|pout (HE): filtre başına keskinlik korunumu ve gürültü azalması}}

{{fig:q1/tire_fig_metrics.png|tire (HE): filtre başına keskinlik korunumu ve gürültü azalması}}

### 1.7 Mean ve Median: Nesne Kenarlarında Karşılaştırma

Aşağıdaki zoom görüntüleri, HE görüntüsünde nesne kenar yoğunluğunun en yüksek olduğu 40×40 pencereden (otomatik ve deterministik seçim; pout için ROI (y, x, h, w) = {{q1.pout.edge_roi_yxhw}}, tire için {{q1.tire.edge_roi_yxhw}}) alınmış ve piksel yapısı görünsün diye NEAREST ile ×6 büyütülmüştür.

{{fig:q1/pout_fig_edge_zoom.png|pout: kenar ROI'sinde Mean ve Median karşılaştırması}}

{{fig:q1/pout_fig_profile.png|pout: kenara dik yoğunluk profili}}

{{fig:q1/tire_fig_edge_zoom.png|tire: kenar ROI'sinde Mean ve Median karşılaştırması}}

{{fig:q1/tire_fig_profile.png|tire: kenara dik yoğunluk profili}}

**Neden farklı davranıyorlar?** Mean filtresi bir lineer alçak geçiren filtredir: kenarın iki yanındaki koyu ve açık pikselleri **ortalar**, dolayısıyla basamak şeklindeki kenar k piksel genişliğinde bir rampaya dönüşür (profilde eğimin yayılması). Median filtresi ise lineer değildir: pencere bir kenarı kestiğinde çoğunluk tarafın değerini **seçer**, ara değer üretmez; bu nedenle basamak kenarlar korunur ve tekil (impulsif) aykırı değerler tamamen elenir.

Sayısal olarak, 3×3 pencerede keskinlik korunumu pout için Mean {{q1.pout.verdict.he.mean_vs_median.k3.mean_sharpness_retained_pct|.1f}}% / Median {{q1.pout.verdict.he.mean_vs_median.k3.median_sharpness_retained_pct|.1f}}%, tire için Mean {{q1.tire.verdict.he.mean_vs_median.k3.mean_sharpness_retained_pct|.1f}}% / Median {{q1.tire.verdict.he.mean_vs_median.k3.median_sharpness_retained_pct|.1f}}%; 5×5 pencerede pout için Mean {{q1.pout.verdict.he.mean_vs_median.k5.mean_sharpness_retained_pct|.1f}}% / Median {{q1.pout.verdict.he.mean_vs_median.k5.median_sharpness_retained_pct|.1f}}%, tire için Mean {{q1.tire.verdict.he.mean_vs_median.k5.mean_sharpness_retained_pct|.1f}}% / Median {{q1.tire.verdict.he.mean_vs_median.k5.median_sharpness_retained_pct|.1f}}% olarak ölçülmüştür. Aynı pencere boyutunda gürültü azalması ise pout için Mean {{q1.pout.verdict.he.mean_vs_median.k3.mean_noise_reduction_pct|.1f}}% / Median {{q1.pout.verdict.he.mean_vs_median.k3.median_noise_reduction_pct|.1f}}% (3×3) şeklindedir.

### 1.8 Ek Deney: Salt & Pepper Gürültüsü

{{fig:q1/pout_fig_saltpepper.png|pout: HE + salt & pepper ve filtre sonuçları}}

{{table:q1_sp:pout}}

{{fig:q1/tire_fig_saltpepper.png|tire: HE + salt & pepper ve filtre sonuçları}}

{{table:q1_sp:tire}}

İmpulsif gürültüde PSNR açısından en iyi filtre pout için **{{q1.pout.verdict.he.saltpepper_best_psnr}}**, tire için **{{q1.tire.verdict.he.saltpepper_best_psnr}}** olmuştur. Mean ve Gaussian filtreler 0/255 değerli aykırı pikselleri komşularına yayarak gri lekeler bırakırken, Median bu pikselleri sıralamada uçlara attığı için doğrudan eler.

### 1.9 Cevap: Hangi filtre gürültüyü temizlerken keskinliği daha iyi korudu?

- **Keskinlik korunumu** açısından en iyi filtre pout'ta **{{q1.pout.verdict.he.best_sharpness_only}}**, tire'da **{{q1.tire.verdict.he.best_sharpness_only}}** olmuştur (Tablo 1.6, zoom ve profil figürleri).
- **Gürültü azaltma ile keskinlik arasındaki denge skoru** en yüksek filtre pout'ta **{{q1.pout.verdict.he.best_label}}**, tire'da **{{q1.tire.verdict.he.best_label}}** olmuştur.
- **İmpulsif (salt & pepper) gürültüde** açık ara en iyi sonuç **{{q1.pout.verdict.he.saltpepper_best_psnr}}** ile alınmıştır (Bölüm 1.8).
- Gaussian filtrede σ arttıkça gürültü azalması artar ancak kenar keskinliği belirgin şekilde düşer (pout: σ=1 → {{q1.pout.filters.he.filters.gauss_s1.sharpness_retained_pct|.1f}}%, σ=2 → {{q1.pout.filters.he.filters.gauss_s2.sharpness_retained_pct|.1f}}% korunum).

**Sonuç:** Nesne kenarlarını koruma ölçütünde Median filtre Mean filtreden üstündür: aynı pencere boyutunda daha yüksek kenar keskinliği korur ve impulsif gürültüyü tamamen temizler. Lineer filtreler (Mean, Gaussian) düz bölgelerdeki ince taneli gürültüyü daha güçlü bastırabilir, fakat bunu kenarları bulanıklaştırma pahasına yapar. Bu nedenle "gürültüyü temizlerken keskinliği koruma" sorusunun cevabı, yukarıdaki görsel kanıtlar ve ölçümlerle birlikte **Median filtre (özellikle 3×3)** olarak verilmiştir.

---

## 2. Soru 2 — Geometrik Dönüşümler ve Kenar Belirleme

**Giriş:** `checkerboard.tif`, boyut {{q2.input.shape}}.

### 2.1 Yöntem: Inverse Mapping ile Affine Dönüşüm

Bir affine dönüşüm, giriş koordinatını (x, y) çıkış koordinatına (x', y') eşleyen 2×3'lük A matrisiyle tanımlanır. **Forward mapping** (her giriş pikselini hedefe göndermek) çıkışta boşluklar (holes) ve çakışmalar üretir. Bu nedenle **inverse mapping** kullanılmıştır: her çıkış pikseli için A⁻¹ ile girişteki (genellikle tamsayı olmayan) kaynak konum hesaplanır ve o konumdaki değer interpolasyonla bulunur. Böylece her çıkış pikseli tam bir kez doldurulur.

- **Rotation (θ = {{q2.angle_deg}}°, merkez etrafında):** $x' = \cos\theta\,(x - c_x) + \sin\theta\,(y - c_y) + c'_x$, $y' = -\sin\theta\,(x - c_x) + \cos\theta\,(y - c_y) + c'_y$. Rotation matrisi: {{q2.cases.rot30_nn.matrix}}, çıkış boyutu {{q2.cases.rot30_nn.shape}}.
- **Shear (yatay, k = {{q2.shear_k}}):** $x' = x + k\,y,\; y' = y$. Matris: {{q2.cases.shear_nn.matrix}}, çıkış boyutu {{q2.cases.shear_nn.shape}}.

{{code:ip.transforms.rotation_matrix}}

{{code:ip.transforms.source_coords}}

**Nearest Neighbor** kaynak konuma en yakın piksel merkezini alır (yuvarlama); **Bilinear** dört komşunun alan ağırlıklı ortalamasını alır. Görüntü dışına düşen komşular 0 kabul edilir (`BORDER_CONSTANT`).

{{code:ip.transforms._sample_nearest}}

{{code:ip.transforms._sample_bilinear}}

**Aliasing'siz referans:** NN ve Bilinear sonuçlarını nesnel olarak değerlendirmek için her çıkış pikselinde 8×8 = 64 alt-piksel konumunda örnekleme yapılıp ortalaması alınan (supersampling, box filtre) bir referans dönüşüm üretilmiştir. Bu, parça parça sabit checkerboard'un ideal alan-ortalamalı dönüşümüne çok yakındır.

**Built-in kontrol:** Manuel dönüşümler `cv2.warpAffine` ile karşılaştırılmıştır. Bilinear'daki küçük farklar, OpenCV'nin alt-piksel konumunu 1/32'ye nicelemesinden (INTER_TAB_SIZE = 32) ve sabit noktalı ağırlıklarından kaynaklanır:

{{table:q2_cv2}}

### 2.2 Dönüşüm Sonuçları

{{fig:q2/fig_transforms.png|Orijinal, shear (NN / Bilinear), rotation 30° (NN / Bilinear) ve aliasing'siz referans}}

### 2.3 Sobel ve Canny

Sobel (3×3, büyüklük $\sqrt{G_x^2 + G_y^2}$, tüm panellerde ortak ölçek) ve Canny ({{q2.canny_thresholds.0}}, {{q2.canny_thresholds.1}}) orijinal ve tüm bozulmuş görüntülere uygulanmıştır:

{{fig:q2/fig_sobel.png|Sobel gradyan büyüklüğü (ortak ölçek)}}

{{fig:q2/fig_canny.png|Canny kenarları}}

### 2.4 NN ve Bilinear: Kenarların Karşılaştırılması

Metrikler, görüntünün tuval sınırından (görüntü/arka plan kenarı) 3 piksel içeride kalan bölgede ve aliasing'siz referansa göre hesaplanmıştır:

{{table:q2_edges}}

- **Sobel MAE (ref.):** Sobel büyüklüğünün referansın Sobel büyüklüğünden ortalama mutlak farkı. Rotation'da NN {{q2.cases.rot30_nn.edges.sobel_mae_vs_reference|.1f}}, Bilinear {{q2.cases.rot30_bilinear.edges.sobel_mae_vs_reference|.1f}}.
- **Sobel fazla enerji %:** Referansta bulunmayan **fazla** Sobel yanıtının (max(|∇I| − |∇I_ref|, 0)) referans enerjisine oranı; doğrudan "false edge" enerjisinin ölçüsüdür. Rotation'da NN %{{q2.cases.rot30_nn.edges.sobel_excess_energy_pct|.1f}}, Bilinear %{{q2.cases.rot30_bilinear.edges.sobel_excess_energy_pct|.1f}}.
- **Yön sapması:** Güçlü gradyanlı piksellerde Sobel gradyan yönünün beklenen yönlerden (rotation için {{q2.cases.rot30_nn.expected_gradient_deg.0|.0f}}° ve {{q2.cases.rot30_nn.expected_gradient_deg.1|.0f}}°) ortalama açısal sapması. Rotation'da NN {{q2.cases.rot30_nn.edges.orientation.mean_abs_dev_deg|.1f}}°, Bilinear {{q2.cases.rot30_bilinear.edges.orientation.mean_abs_dev_deg|.1f}}°.

{{fig:q2/fig_orientation_hist.png|Sobel gradyan yönü dağılımı: NN, Bilinear ve referans}}

### 2.5 NN Aliasing'in Sobel'de Oluşturduğu Hatalı Kenarlar (Zoom)

Aşağıdaki kırpmalar **sabit ROI koordinatlarıyla** programatik olarak alınmış ve piksellerin görünür kalması için NEAREST ile ×8 büyütülmüştür. Her figürde satırlar NN / Bilinear / Referans, sütunlar görüntü / Sobel / Canny'dir. Canny panelinde yeşil = referansla uyumlu kenar, kırmızı = referanstan 1 pikselden uzak (false) kenar, mavi = kaçırılan referans kenarıdır.

{{fig:q2/fig_roi_map_rot30.png|Rotation 30° (NN) üzerinde ROI konumları}}

{{fig:q2/fig_zoom_rot30_roi1.png|Rotation 30°, ROI 1: NN / Bilinear / referans}}

{{fig:q2/fig_zoom_rot30_roi2.png|Rotation 30°, ROI 2: NN / Bilinear / referans}}

{{fig:q2/fig_zoom_shear_roi1.png|Shear, ROI 1: NN / Bilinear / referans}}

**Açıklama.** NN interpolasyonu her çıkış pikseline en yakın kaynak pikselin değerini kopyalar ve ara değer üretmez. 30° eğik bir karo kenarı bu nedenle yatay ve dikey basamaklardan oluşan bir merdiven (staircase / jaggies) şeklinde örneklenir: kenar boyunca birkaç piksel yatay ilerler, sonra 1 piksel dikey zıplar. Sobel operatörü yerel türev aldığı için bu basamakların **her birini ayrı bir yatay veya dikey mini kenar** olarak algılar:

1. Gradyan yönü, gerçek kenarın normali olan {{q2.cases.rot30_nn.expected_gradient_deg.0|.0f}}° / {{q2.cases.rot30_nn.expected_gradient_deg.1|.0f}}° yerine 0° ve 90°'ye kayar (yön histogramında NN eğrisinin eksenlerde biriken bileşenleri; güçlü piksellerin %{{q2.cases.rot30_nn.edges.orientation.pct_off_axis_gt10deg|.1f}}'i beklenen yönden 10°'den fazla sapar, Bilinear'da %{{q2.cases.rot30_bilinear.edges.orientation.pct_off_axis_gt10deg|.1f}}).
2. Kenar şiddeti basamak köşelerinde tam kontrastla sıçrar, basamakların ortasında ise farklı davranır; Sobel çıktısı düzgün bir çizgi yerine kesikli ve kalınlığı değişen bir desen gösterir. Bu, gerçek sahnede bulunmayan **hatalı (false) kenar** enerjisidir (fazla enerji: NN %{{q2.cases.rot30_nn.edges.sobel_excess_energy_pct|.1f}}, Bilinear %{{q2.cases.rot30_bilinear.edges.sobel_excess_energy_pct|.1f}}).
3. Canny, inceltme (non-maximum suppression) ve histerezis sonrası bu basamakları takip eder; kenar zinciri gerçek eğik çizgi yerine merdiven biçimini izler (Canny konum uyumsuzluğu: NN %{{q2.cases.rot30_nn.edges.canny_exact_mismatch_pct|.1f}}, Bilinear %{{q2.cases.rot30_bilinear.edges.canny_exact_mismatch_pct|.1f}}).

Bilinear interpolasyon ise kenarı geçen piksellere ara gri değerler verir (doğal anti-aliasing); Sobel yanıtı kenar boyunca sürekli kalır ve yönü gerçek kenar normaline yakındır. Bunun bedeli, kenarın yaklaşık 1 piksel yumuşamasıdır. Aynı etki shear görüntüsünde de görülür (Sobel MAE: NN {{q2.cases.shear_nn.edges.sobel_mae_vs_reference|.1f}}, Bilinear {{q2.cases.shear_bilinear.edges.sobel_mae_vs_reference|.1f}}).

---

## 3. Soru 3 — Sıfırdan Bilinear Resize ve Laplacian Keskinleştirme

**Giriş:** `cameraman.tif`, boyut {{q3.input.shape}}, ölçek {{q3.scale}} → çıkış {{q3.resize.out_shape}}.

### 3.1 Bilinear Interpolation Matematiği

Her çıkış pikseli (x, y) için girişteki kaynak konum **inverse mapping** ile bulunur. Piksel merkezleri hizalı (half-pixel) kural, oran = giriş boyutu / çıkış boyutu = {{q3.resize.ratio_x|.6f}} olmak üzere:

$$ s_x = (x + 0.5)\cdot oran - 0.5,\qquad s_y = (y + 0.5)\cdot oran - 0.5 $$

Kaynak konum [0, N−1] aralığına sıkıştırılır (kenarlarda replicate). Dört komşu ve kesirli uzaklıklar:

$$ x_0 = \lfloor s_x \rfloor,\; x_1 = \min(x_0 + 1, W-1),\; d_x = s_x - x_0 \qquad (y_0, y_1, d_y \text{ benzer}) $$

$$ out = (1-d_x)(1-d_y)\,I(x_0,y_0) + d_x(1-d_y)\,I(x_1,y_0) + (1-d_x)\,d_y\,I(x_0,y_1) + d_x\,d_y\,I(x_1,y_1) $$

Dört ağırlığın toplamı 1'dir; her ağırlık, kaynak noktanın **karşı köşedeki** komşuya olan uzaklığıyla orantılı alanı temsil eder.

### 3.2 Kod: İç İçe For Döngüleri ile Bilinear Resize

Raporda istenen sürüm aşağıdadır: dış döngü çıkış satırları (y), iç döngü çıkış sütunları (x) üzerinde döner ve dört komşu ile dört ağırlık ayrı değişkenler olarak açıkça hesaplanır.

{{code:ip.resize.bilinear_resize_loops}}

Aynı aritmetiği NumPy broadcasting ile yapan bir ikiz (`bilinear_resize_vec`) de yazılmış ve döngü sürümüyle **bit düzeyinde aynı** olduğu doğrulanmıştır (cameraman üzerinde: {{q3.resize.loops_equals_vec_exact}}). Döngü sürümünün çalışma süresi: {{q3.resize.loops_seconds|.2f}} s.

**Sayısal örnek** (çıkış pikseli (y, x) = {{q3.resize.worked_example.out_pixel_yx}}): $s_y$ = {{q3.resize.worked_example.src_y|.4f}}, $s_x$ = {{q3.resize.worked_example.src_x|.4f}} → $x_0$ = {{q3.resize.worked_example.x0}}, $x_1$ = {{q3.resize.worked_example.x1}}, $y_0$ = {{q3.resize.worked_example.y0}}, $y_1$ = {{q3.resize.worked_example.y1}}, $d_x$ = {{q3.resize.worked_example.dx|.4f}}, $d_y$ = {{q3.resize.worked_example.dy|.4f}}.

{{table:q3_example}}

### 3.3 Kod: İç İçe For Döngüleri ile Laplacian (High-pass) Filtre

4-komşu Laplacian kernel'i K = [[0, 1, 0], [1, −4, 1], [0, 1, 0]] kullanılmıştır:

$$ L(y,x) = I(y-1,x) + I(y+1,x) + I(y,x-1) + I(y,x+1) - 4\,I(y,x) $$

Görüntü önce `reflect101` modunda 1 piksel padding'lenir; ardından dört iç içe döngü (y, x, kernel satırı i, kernel sütunu j) ile korelasyon toplamı hesaplanır. Hesap `float64` ile yapıldığı için negatif değerler ve 255'i aşan değerler korunur (uint8 taşması yoktur). Merkez katsayısı negatif olduğundan keskinleştirme $S = I - c\cdot L$ (c = 1) ile yapılır.

{{code:ip.filters.laplacian_loops}}

{{code:ip.filters.sharpen}}

### 3.4 Sonuçlar

{{fig:q3/fig_resize_compare.png|Orijinal, manual Bilinear (half-pixel), cv2.resize ve manual Bilinear (naive)}}

{{fig:q3/fig_resize_zoom.png|Aynı sahne bölgesinin zoom karşılaştırması}}

{{fig:q3/fig_laplacian_sharpen.png|Resize sonucu, Laplacian yanıtı ve keskinleştirilmiş görüntüler}}

{{fig:q3/fig_sharpen_zoom.png|Keskinleştirme öncesi/sonrası zoom}}

Laplacian yanıtı [{{q3.laplacian.range.0|.0f}}, {{q3.laplacian.range.1|.0f}}] aralığındadır ve piksellerin %{{q3.laplacian.pct_negative|.1f}}'i negatiftir. Keskinleştirme sonrası piksellerin %{{q3.sharpen.pct_clipped_low|.2f}}'si 0'ın altına, %{{q3.sharpen.pct_clipped_high|.2f}}'si 255'in üstüne çıktığı için clip'lenmiştir.

### 3.5 Manuel ve Built-in Karşılaştırması

Built-in referanslar: `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` (aynı `dsize`), `cv2.filter2D` (aynı kernel, `float64`, `BORDER_REFLECT_101`) ve `cv2.Laplacian(ksize=1)`.

{{table:q3_compare}}

{{fig:q3/fig_resize_diff_heatmap.png|Resize fark haritaları: half-pixel kural (solda) ve naive kural (sağda)}}

{{fig:q3/fig_laplacian_diff_heatmap.png|Laplacian ve tam pipeline fark haritaları}}

### 3.6 Farkların Nedenleri

1. **Piksel merkezi kuralı (half-pixel offset).** OpenCV kaynak konumu $s = (d + 0.5)\cdot oran - 0.5$ ile hesaplar; yani piksellerin **merkezlerini** hizalar. Naive $s = d\cdot oran$ eşlemesi ise piksel **köşelerini** hizalar ve görüntüyü sola-yukarı yaklaşık 0.5·(1 − 1/1.7) ≈ 0.21 giriş pikseli kaydırır. Bu kayma kenarlarda büyük farka dönüşür: naive kural ile maksimum fark {{q3.resize.naive_vs_cv2.max_abs_diff}}, ortalama {{q3.resize.naive_vs_cv2.mean_abs_diff|.2f}}; fark haritasında (sağda) farkın nesne kenarlarında yoğunlaştığı görülür. Half-pixel kural kullanıldığında maksimum fark {{q3.resize.half_pixel_vs_cv2.max_abs_diff}} gri seviyeye iner (PSNR {{q3.resize.half_pixel_vs_cv2.psnr_db|.2f}} dB).
2. **Yuvarlama / sabit noktalı aritmetik.** Kalan ±1'lik fark (piksellerin %{{q3.resize.half_pixel_vs_cv2.pct_pixels_differ|.1f}}'inde), OpenCV'nin `uint8` resize'da ağırlıkları 11 bitlik tamsayılara (INTER_RESIZE_COEF_BITS = 11, ölçek 2048) yuvarlayıp tamsayı aritmetiği kullanmasından kaynaklanır. Manuel sürüm ağırlıkları `float64` ile tam hesaplar ve sonucu yalnızca en sonda yuvarlar. Float manuel sonuç ile OpenCV `uint8` sonucu arasındaki maksimum fark {{q3.resize.half_pixel_float_vs_cv2.max_abs_diff|.3f}}'tür; yani fark tamamen son yuvarlama adımındadır. Fark haritası (soldaki) yapısız, gürültü benzeri bir 0/1 dağılımı gösterir; sistematik bir kayma yoktur.
3. **Border (padding) modu.** Laplacian'da manuel sürüm ve OpenCV aynı `reflect101` modunu kullandığında fark **tam olarak {{q3.laplacian.loops_vs_filter2d_float.max_abs_diff}}**'dır. Farklı modlar yalnızca görüntünün en dış 1 piksellik çerçevesini etkiler:

{{table:q3_border}}

4. **uint8 taşması (overflow).** Laplacian negatif değerler ve 255'ten büyük değerler üretir. `uint8` çıktılı `cv2.filter2D` negatifleri 0'a **doyurur** (saturation), NumPy `uint8` aritmetiği ise mod 256 **sarar** (wrap-around). Her iki durumda da float sonuçtan büyük sapmalar oluşur (tablodaki son iki satır) ve doyurulmuş Laplacian ile yapılan keskinleştirme yanlış sonuç verir (fark: maksimum {{q3.overflow.sharpen_with_saturated_vs_float.max_abs_diff}}). Bu yüzden tüm hesaplar `float64` ile yapılmıştır.

{{fig:q3/fig_overflow.png|uint8 taşmasının etkisi: float (doğru), doygunluk, wrap-around ve yanlış keskinleştirme}}

{{fig:q3/fig_border_zero_vs_reflect.png|Zero padding ile reflect101 arasındaki fark yalnızca çerçevede}}

Tam pipeline'da (manuel resize → manuel Laplacian → keskinleştirme ile cv2.resize → cv2.Laplacian → keskinleştirme) kalan fark (maksimum {{q3.sharpen.manual_pipeline_vs_cv2_pipeline.max_abs_diff}}), resize adımındaki ±1 farkların Laplacian'ın merkez katsayısı (−4) ve komşu katsayılarıyla büyütülmesinden kaynaklanır; aynı girişe uygulandığında keskinleştirme farkı {{q3.sharpen.manual_vs_builtin_same_input.max_abs_diff}}'dır.

---

## 4. Özet

| Soru | Ana bulgu | Kanıt |
| --- | --- | --- |
| Q1 | Manuel Contrast Stretching ve HE her iki görüntüde çıktıyı [0, 255] aralığına yaydı; HE, `cv2.equalizeHist` ile en fazla {{q1.pout.enhance.he_vs_cv2_equalizeHist.max_abs_diff}} (pout) / {{q1.tire.enhance.he_vs_cv2_equalizeHist.max_abs_diff}} (tire) farklı. Kenar keskinliğini en iyi koruyan filtre: {{q1.pout.verdict.he.best_sharpness_only}} (pout), {{q1.tire.verdict.he.best_sharpness_only}} (tire). | Bölüm 1.3–1.9 |
| Q2 | NN rotation'da Sobel fazla (false) kenar enerjisi %{{q2.cases.rot30_nn.edges.sobel_excess_energy_pct|.1f}}, Bilinear'da %{{q2.cases.rot30_bilinear.edges.sobel_excess_energy_pct|.1f}}; NN basamakları gradyan yönünü eksenlere kaydırıyor. | Bölüm 2.4–2.5 |
| Q3 | Half-pixel kuralıyla manuel Bilinear, cv2.resize ile en fazla {{q3.resize.half_pixel_vs_cv2.max_abs_diff}} farklı (PSNR {{q3.resize.half_pixel_vs_cv2.psnr_db|.2f}} dB); nested-loop Laplacian, cv2.filter2D ile birebir aynı (fark {{q3.laplacian.loops_vs_filter2d_float.max_abs_diff}}). | Bölüm 3.5–3.6 |
