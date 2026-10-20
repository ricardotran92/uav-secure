# Handoff: tái hiện Tang et al. (TMC 2025) đã chốt ở R15; bước tiếp là giải pháp mới (trạng thái 2026-10-20)

Dành cho: người (hoặc phiên làm việc) tiếp quản dự án. Đọc xong file này là đủ để chạy lại, hiểu các quyết định đã chốt và bắt đầu phần giải pháp mới. Chi tiết từng thay đổi nằm ở [MODEL_HISTORY.md](MODEL_HISTORY.md) (bảng xếp hạng, bảng sàng lọc 🥇🥈🥉, bảng so với paper), bảng tham số ở [PARAMETERS.md](PARAMETERS.md).

## 1. Tình trạng một đoạn

Code PyTorch tái hiện toàn bộ pipeline của paper đã chạy end-to-end trên GPU: GNN beamforming, SAC deployment, so sánh với MLP, OPT và heuristic, báo cáo tự động và animation. Người dùng **chốt R15 làm kết quả tái hiện** (2026-10-20). Điểm so với paper của R15 là **0.095** khi lấy trung bình 3 seed SAC, 0.083 với 1 seed (`scripts/paper_scorecard.py`, càng thấp càng khớp; R14 0.0985, R13 0.109; bản Table I nguyên văn khoảng 2.4). 16 thông số Table I và 5 lớp GCN giữ nguyên (kiểm tự động, 0 lệch). 36 test đều qua. **Việc tiếp theo là đề xuất và kiểm chứng một giải pháp mới** (mục 5), để lấy điểm bonus và hướng tới publish. **Chưa có gì được push lên GitHub**: mọi commit chỉ ở máy local.

## 2. Kết quả tái hiện R15 so với paper

R15 = Graph-DRL của paper với các chi tiết paper không công bố được chọn bằng scorecard. Phần kiến trúc gồm GNN 5 lớp, lớp ẩn 768, LayerNorm và kết nối tắt trong mỗi lớp GCN, aggregate max trên N(k) ∪ {k}, đặc trưng nút không phụ thuộc công suất test. Phần kênh dùng cách hiểu đã hiệu chỉnh (mục 4). Thư mục: `outputs/runs_calibrated/`; seed SAC thêm ở `outputs/sac_seeds/r15_seed1|2`. Số paper đọc từ hình nên có sai số khoảng ±5%.

**Bằng hoặc vượt paper (trong sai số đọc hình):**

| Chỉ số | R15 | Paper |
|---|---|---|
| Graph-DRL tại vị trí học được, Fig. 8 | 23.7 bps/Hz | 23 |
| GNN K=8 so với OPT / GNN K=11 | 0.94 / 16.2 | 0.95 / 16.1 |
| GNN K=8 | 14.0 | 14.8 (−5%) |
| GNN vs công suất 10 / 20 / 30 dBm | 2.2 / 6.7 / 14.0 | 2.0 / 7.0 / 14.6 |
| OPT vs công suất 0 / 20 / 40 dBm; OPT K=8 | 0.94 / 7.5 / 26.4; 15.0 | 1.0 / 7.5 / 28.0; 15.5 |
| Điểm tối ưu Fig. 7 cách điểm của paper (56, 140) | 6.3 m (3 seed GNN: 7.2 / 6.3 / 6.3) | 0 |
| SAC dừng cách điểm tối ưu (3 seed) | 3.4 / 9.1 / 2.1 m | khoảng 3 m |
| Loss transfer / re-train K=12, Fig. 5 | −16.8 / −16.1 | −18.0 / −17.7 |

**Còn kém paper:**

| Chỉ số | R15 | Paper |
|---|---|---|
| GNN K=1 (so với OPT) | 2.93 (0.80) | 3.86 (0.91) |
| GNN 40 dBm (so với OPT) | 21.1 (0.80) | 23.8 (0.85) |
| Loss GNN cuối, Fig. 4 | −13.6 | −14.9 |
| Reward SAC cuối, Fig. 6 (3 seed) | 4010-4108 / 23.5 bps/Hz | 4400 / 25 |
| Lợi thế Graph-DRL so với heuristic tốt nhất, Fig. 8 | +15% (heuristic đạt 0.69-0.87 Graph-DRL) | +64% (0.39-0.61) |
| MLP K=8 (baseline, không làm yếu đi) | 9.63, mạnh hơn | 5.4 |

**Bản thay thế R15b** (`outputs/archive_r15b/`): R15 với khoảng cách tối thiểu 25 m. Bản này tốt nhất ở Fig. 4/5/6/8/9a/10 (reward 4205-4298, lợi thế Graph-DRL +23-26%), nhưng Fig. 9b kém hẳn. Trung bình 3 seed được 0.096, gần như hòa với R15.

**Khi trình bày phải nói rõ:**
1. Các chỉ số khớp dựa trên cách hiểu kênh đã hiệu chỉnh, không phải giá trị paper công bố.
2. OPT là bộ giải thay thế của mình (gradient đa khởi tạo), **không có artificial noise (AN)**, trong khi OPT [45] của paper có precoding + AN.
3. Cặp user-eve của layout Fig. 7 là suy luận.
4. MLP và heuristic của mình mạnh hơn paper, nên lợi thế của GNN trông nhỏ hơn. Đó không phải GNN thua.
5. Nên trích điểm 3 seed (0.095), vì điểm 1 seed có phần may ở Fig. 7 SAC.

## 3. Các quyết định đã chốt

| Quyết định | Lý do |
|---|---|
| **R15 đóng băng làm kết quả tái hiện và baseline cho giải pháp mới** (2026-10-20) | người dùng chốt; mọi thí nghiệm sau không được làm đổi kết quả R15 |
| Giữ nguyên mọi giá trị Table I; chỉ đổi chi tiết paper không công bố | yêu cầu của người dùng; kiểm tự động bằng `export_docs.py` |
| Không làm yếu MLP, OPT hay heuristic để số liệu đẹp hơn | tránh gian lận số liệu; thầy hỏi một câu là lộ |
| Nhận một thay đổi chỉ khi điểm scorecard giảm (và điểm tối ưu Fig. 7 trung vị ≤ 20 m); khi hòa, chọn cách đọc sát nghĩa đen paper | tiêu chí khách quan |
| So các bản ứng viên bằng điểm trung bình 3 seed SAC (C47) | 1 seed SAC lệch điểm tới 0.013 |
| Lịch LR cosine bắt đầu từ 0.005; kênh đã hiệu chỉnh; giữ cặp U5-E5, U6-E6 | người dùng duyệt các ngày 2026-10-14 đến 10-17 |
| Điểm tối ưu Fig. 7 tìm bằng GNN (ràng buộc (24) của bài toán (23)-(25)) | bám đúng paper |
| Không push GitHub khi chưa được hỏi | quy ước của phiên làm việc |

Các hướng **đã thử và loại** (mục C-xx trong MODEL_HISTORY.md, kèm số liệu) gồm:
- **Kênh và đầu vào:** paper SNR (nhiễu +30 dB), nén log đầu vào, eve cách user 20 ± 5-10 m, khoảng cách tối thiểu 1/10/20/50/100 m.
- **Cách train GNN:** UAV rải đều hoặc lệch quanh tâm khi train, khởi tạo MRT, SGD.
- **Kiến trúc GNN:** f_out 3 lớp; lớp ẩn 768/1024 khi chưa có LayerNorm; chỉ LayerNorm, chỉ kết nối tắt, hoặc LayerNorm chỉ ở f_out; quy tắc K=1 "self" khi suy luận; 300 mini-batch mỗi episode.
- **SAC:** bước bay 1 m.

## 4. Cấu hình R15 và cách đọc

Giá trị mặc định nằm ở `configs/default.yaml`; profile chính chồng thêm `configs/calibrated.yaml`. Mỗi giá trị không phải của paper đều có chú thích `# assumption`.

- **Của paper, giữ nguyên:**
  - hệ thống: vùng 200×200 m, 8 user/eve, 8 anten, độ cao 100 m, Rician 10 dB, 1 Hz, 1 W, nhiễu 1.2e-13 W;
  - GNN: lr 0.005, 300 episode, batch 512, 5 lớp;
  - SAC: lr 3e-4, 500 episode, mini-batch 64, buffer 1e6, discount 0.99, khoảng hành động [-1, 1].
- **Hiệu chỉnh (không phải giá trị paper):** path loss `30 + 22 ln(d)`, khoảng cách ngang, khoảng cách tối thiểu 30 m. Paper chỉ ghi "log(d)". Cách hiểu thông thường (log10, 3-D) cho bản đồ rate gần như phẳng, không tái tạo được Fig. 7/8.
- **Giả định GNN:**
  - lớp ẩn 768;
  - `gnn.aggregate_self=true`, `gnn.layer_norm=layer`, `gnn.residual=true`;
  - `gnn.input_power_scaling=false`: đặc trưng nút cố định ở P^max, công suất test chỉ vào qua chuẩn hóa W;
  - 150 mini-batch mỗi episode, LR cosine, Adam, cắt gradient 1.0;
  - train với UAV đúng tại tâm vùng (câu "this is the setup to train the GNN model" của paper).
- **Giả định SAC:** 2 gradient step mỗi bước, bước bay 2.5 m, 175 bước mỗi episode, trạng thái nén log, không scale reward, chọn policy tốt nhất qua đánh giá 3 episode.
- **Tìm điểm tối ưu:** dùng GNN, 128 mẫu fading.
- **Layout Fig. 7:** số hóa từ hình (±1 m), `configs/layouts/paper_fig7.yaml`.

## 5. Bước tiếp: giải pháp mới (bonus và hướng publish)

### 5.1 Mục tiêu

Yêu cầu môn học (`source_materials/course-requirements.md`): báo cáo cuối kỳ (Word/PDF) và slide bằng tiếng Anh. Phần bonus: *"If you propose a new solution in your final report and include the validation results, you can get the 10 bonus points."*

Người dùng ưu tiên một **giải pháp mới thật sự**, nghĩa là đổi bài toán hoặc phương pháp chứ không chỉ tune lại. Lý do: tune lại thì không đủ để publish ở hội nghị hay tạp chí. Tiêu chí kiểm chứng: so với Graph-DRL của paper (bản tái hiện R15) và các baseline trên cùng tập test và cùng seed, và chỉ ra một khía cạnh mà phương pháp của paper không xử lý được.

### 5.2 Những giả định yếu của paper (động lực cho giải pháp mới)

1. **Biết CSI hoàn hảo của eavesdropper.** Eve là thiết bị nghe lén thụ động, không phát tín hiệu, vậy mà h_E,k lại là đầu vào trực tiếp của GNN (eq. (12)). Đây là điểm yếu dễ thuyết phục reviewer nhất.
2. **GNN không dùng artificial noise.** OPT [45] mà paper so sánh có precoding + AN, còn GNN chỉ sinh w_k. Ở công suất cao, khi eve nhận tốt, AN thường là cách hiệu quả nhất. Đây cũng là chỗ GNN của mình kém paper nhất (40 dBm: 0.80 so với OPT).
3. **Topology tĩnh.** User và eve đứng yên, nên chạy exhaustive search một lần là ra điểm tối ưu (pipeline hiện tại đã làm việc này), còn SAC chỉ đạt 96-99% rate tại điểm đó (3 seed R15). DRL chưa có lý do tồn tại rõ ràng.
4. Độ cao cố định 100 m, một UAV, GNN yếu khi K=1.

### 5.3 Kiểm tra tài liệu nhanh (2026-10-20, chưa đầy đủ)

- [SecGNN, IEEE TVT](https://cell-devs-02.sce.carleton.ca/publications/2023/IAWBBW23/[2n].pdf): GNN ánh xạ kênh thành beamforming + AN, train không nhãn; **không có UAV**. Như vậy "GNN + AN" một mình **không mới**.
- [GNN dị thể hai tầng cho RIS-PLS (arXiv 2603.14655, 2026)](https://arxiv.org/html/2603.14655v1): sinh beamforming và AN; không có UAV.
- [GNN cho bảo mật đa vệ tinh (JSAC 2026)](https://pure.bit.edu.cn/en/publications/gnn-based-secrecy-rate-optimization-in-multi-satellite-collaborat/): có CSI không hoàn hảo; không có UAV.
- [DRL cho truyền bảo mật UAV-RIS](https://pure.qub.ac.uk/en/publications/learning-based-reliable-and-secure-transmission-for-uav-ris-assis/): có AN, vị trí UAV, CSI không hoàn hảo, nhưng dùng DQN, không dùng GNN.
- Paper gốc: [arXiv 2504.01446](https://arxiv.org/pdf/2504.01446).
- **Chưa thấy:**
  - bài kết hợp UAV + GNN + AN + bất định CSI/vị trí của eve;
  - bài tối ưu vị trí UAV bằng gradient xuyên qua GNN beamformer.
- **Phải làm review tài liệu đúng cách** (IEEE Xplore, Google Scholar, các bài trích dẫn Tang et al.) trước khi viết là "mới". Ba lần search ở trên chưa đủ để khẳng định.

### 5.4 Đề xuất khuyến nghị: Robust AN-aided Graph-DRL khi vị trí eavesdropper không chắc chắn

Ý tưởng một câu: UAV chỉ biết vị trí ước lượng của eve (sai số tới r mét), không có h_E,k. GNN sinh cả beamforming lẫn AN và được train để tối đa secrecy rate kỳ vọng (hoặc giảm secrecy outage) trên phân bố vị trí eve. Deployment tối ưu theo cùng mục tiêu bền vững đó.

**(a) Mô hình bất định**
- Vị trí thật của eve là `q̂_E,k + δ`, với δ đều trong đĩa bán kính r (hoặc Gauss). Kênh thật lấy mẫu từ vị trí thật.
- Đầu vào mạng chỉ gồm phần LoS (steering vector cùng path loss) tính từ `q̂_E,k`, cộng thêm r nếu cần.
- Chỉ cần lấy mẫu `scenario.eves` nhiều lần rồi gọi `channels(...)`.
- Trường hợp r = 0 và có đầu vào h_E hoàn hảo phải lùi về đúng bài toán của paper.

**(b) GNN có AN**
- Thêm một đầu ra AN: một vector z (hoặc v hạng thấp) lấy từ pooling toàn đồ thị, cùng tỉ lệ chia công suất ρ học được: ‖W‖² = ρP, ‖z‖² = (1−ρ)P.
- SINR có thêm `|h^H z|²` ở mẫu số của cả user lẫn eve.
- Lưu ý: N = K = 8 nên **không có null-space** của kênh user để đặt AN vào. Mạng phải tự học khi nào nên dùng AN; khi K < N mới có null-space.

**(c) Loss bền vững**
- −E_δ[R_sec] với M mẫu δ cho mỗi instance.
- Phương án mạnh hơn: CVaR_α của R_sec, hoặc thêm phạt secrecy outage P(R_sec,k < R_th).

**(d) Deployment**
- Reward SAC là secrecy rate bền vững.
- Thêm một phương pháp so sánh mới: **tối ưu vị trí bằng gradient xuyên GNN** (q_U → `channels` → GNN → R_sec, toàn bộ là torch) với đa khởi tạo. So với SAC và exhaustive search về chất lượng và thời gian. Có thể tách thành đóng góp thứ hai.
- Cần kiểm tra trước: gradient theo q_U có đi qua `rician_channels` không. Khoảng cách tối thiểu (clamp) sẽ chặn gradient khi UAV đứng gần node.

**Chỗ sửa trong code** (mọi khóa mới mặc định tắt):

| File | Việc |
|---|---|
| `src/uavsec/secrecy.py` | `sinr_rates` / `secrecy_rates` thêm tham số AN (mặc định None = như hiện tại); hàm chia công suất cạnh `normalize_power` |
| `src/uavsec/models/gnn.py` | đầu AN (pooling toàn đồ thị) và ρ; tùy chọn đầu vào eve chỉ gồm LoS ước lượng |
| `src/uavsec/models/base.py` | `features` và `forward` trả thêm AN |
| `src/uavsec/channel.py` | lấy mẫu vị trí eve quanh vị trí ước lượng; hàm sinh kênh LoS ước lượng |
| `src/uavsec/training/beamformer_trainer.py` | loss bền vững (M mẫu eve, kỳ vọng hoặc CVaR) |
| `src/uavsec/baselines/opt_beamforming.py` | **OPT có AN** (baseline bắt buộc, gần OPT [45] hơn), chạy với CSI ước lượng và với CSI hoàn hảo |
| `src/uavsec/envs/uav_env.py`, `src/uavsec/baselines/deployment.py` (`expected_rate`, `exhaustive_search`) | reward và bản đồ rate tính theo mục tiêu bền vững |
| `src/uavsec/evaluation/beamforming.py`, `scripts/exp_*.py` | sweep mới theo r; chỉ số secrecy outage |

**Kế hoạch kiểm chứng** (3 seed GNN × 3 seed SAC, cùng tập test):
- **Phương pháp so sánh:**
  - Graph-DRL của paper (R15; được cho CSI eve hoàn hảo, hoặc chỉ có ước lượng, nhưng chấm trên kênh thật);
  - OPT không có AN và OPT có AN, mỗi cái chạy với CSI ước lượng và với CSI hoàn hảo (làm cận trên);
  - MLP;
  - các heuristic deployment.
- **Hình:**
  - secrecy rate và secrecy outage theo r (0, 5, 10, 20 m);
  - theo công suất (−20..40 dBm) và theo K;
  - CDF;
  - Fig. 7/8 với eve bất định (vị trí tối ưu bền vững so với vị trí tối ưu khi tưởng CSI hoàn hảo);
  - thời gian chạy (GNN so với OPT).
- **Ablation:** bỏ AN, bỏ loss bền vững, bỏ deployment bền vững.
- **Kiểm tra lùi:** ở r = 0 với đầu vào hoàn hảo, phương pháp mới phải đạt ít nhất bằng R15.

**Thứ tự làm và ước lượng:**
1. AN trong `secrecy.py` cùng OPT-AN, cộng GNN-AN với CSI hoàn hảo: vài ngày code, 1-2 vòng sàng lọc. Kết quả mong đợi: thu hẹp khoảng 40 dBm.
2. Bất định vị trí eve cùng loss bền vững: vài ngày code, sweep theo r.
3. Deployment bền vững (SAC và gradient), rồi chạy đầy đủ, rồi viết báo cáo.

Mỗi giai đoạn tự nó đã có kết quả trình bày được nếu hết thời gian.

### 5.5 Hướng thay thế

- **B. Deployment khả vi qua GNN** (thay SAC hoặc khởi động SAC).
  - Novelty: chưa thấy trong tài liệu.
  - Công sức thấp, vì code đã gần đủ.
  - Rủi ro: bản đồ rate nhiều cực trị địa phương (cần đa khởi tạo), và clamp khoảng cách chặn gradient.
  - Hợp làm đóng góp phụ trong 5.4.
- **C. User và eve di chuyển** (topology thay đổi theo thời gian; SAC theo dõi được, exhaustive search phải chạy lại mỗi bước).
  - Novelty: cho DRL lý do tồn tại, sửa được điểm yếu 3 của paper.
  - Công sức trung bình: môi trường mới, trạng thái có vị trí và vận tốc.
  - Rủi ro: tốn thời gian train SAC.
- **D. UAV 3D (độ cao) hoặc nhiều UAV.**
  - Novelty: trung bình, vì UAV 3D đã có nhiều bài.
  - Công sức: nhiều UAV thì lớn (đồ thị dị thể).
  - Rủi ro: lệch khỏi khung paper.

### 5.6 Quy tắc khi làm giải pháp mới

- **Không làm đổi R15:**
  - khóa mới mặc định tắt;
  - profile riêng (ví dụ `configs/proposed.yaml` chồng lên `default.yaml` + `calibrated.yaml`), thư mục `outputs/proposed_*`;
  - nên đặt git tag `r15-reproduction` trên commit hiện tại trước khi sửa code (chưa làm);
  - chạy `python -m pytest` để chắc test cũ vẫn qua.
- **So công bằng:**
  - cùng tập test, cùng layout Fig. 7, cùng seed;
  - không làm yếu baseline;
  - OPT-AN phải được tối ưu tử tế, ngang OPT hiện tại.
- **Ghi lại:** dùng `docs/history.json` với dãy ID mới (P01, P02, ...). Bảng sàng lọc dùng khóa `screening` (`scripts/summarize_screening.py`). `export_docs.py` sinh lại tài liệu.
- **Báo cáo hai chế độ:** CSI hoàn hảo (cho thấy không mất gì so với paper) và CSI bất định (cho thấy lợi ích).

## 6. Việc còn dở khác

- **Sản phẩm nộp môn học:** báo cáo mô phỏng (Word/PDF) và slide tiếng Anh, chưa có.
  - Nguồn: `outputs/runs_calibrated/report/report.md` (kèm hình PNG/PDF), animation `outputs/runs_calibrated/deployment_*/animation/uav_trajectory.gif`, các bảng trong MODEL_HISTORY.md.
  - Báo cáo nên có hai phần: tái hiện (R15) và giải pháp mới.
- Push các commit local lên `origin` (cần người dùng đồng ý và có quyền).
- `outputs/runs` (profile Table I nguyên văn) đã lỗi thời từ R14, vì `default.yaml` đổi. Chạy lại nếu cần cho báo cáo (khoảng 8 giờ với lớp ẩn 768).
- Fig. 11 (thời gian chạy) chưa số hóa và chưa chấm. GNN nhanh hơn OPT khoảng 1500 lần, nhưng thời gian OPT của mình không tăng theo K như paper.
- R15a mới có 1 seed SAC (không quan trọng).

## 7. Cách chạy lại

- **Môi trường local:** Python 3.10 (`pip install -e ".[dev]"`), `python -m pytest`. Chạy thử nhanh mọi script bằng `--quick`.
- **Server:** `ricardo@140.123.104.113`, key `C:\Users\Ricardo\.ssh\id_ed25519`, conda env `uavsec` (torch 2.5.1+cu121; **không nâng torch**, driver 535 không chạy được bản mới hơn), GPU 2 và 3 (1080 Ti). Hiện không có phiên tmux nào đang chạy.
- **Đẩy code / kéo dữ liệu:** `bash scripts/server/sync.sh push` (tar qua ssh). Kéo dữ liệu bằng `ssh ... "tar -czf - outputs/<thư mục>" | tar -xzf -`. **Không in gì khác ra stdout trong cùng lệnh ssh**, vì nó lẫn vào luồng tar và làm hỏng file.
- **Chạy đầy đủ một profile:** `GPU=2 bash scripts/server/run_all.sh --config configs/default.yaml configs/calibrated.yaml --set output_dir=outputs/runs_xxx`. Mất khoảng 6-8 giờ với lớp ẩn 768. `scripts/server/run_deployment.sh` chỉ chạy Phase 4-5.
- **Sàng lọc GNN:** `GPU=2 bash scripts/server/screen_gnn.sh outputs/studies_xxx/<tên> <khóa>=<giá trị> ...` (3 seed và đánh giá Fig. 9a/9b, khoảng 1.5-4 giờ). Tổng hợp bằng `python scripts/summarize_screening.py outputs/studies_xxx --json rows.json`.
- **Thêm seed SAC:** `python scripts/exp_deployment.py --config ... --gnn outputs/<runs>/convergence_*/ckpt/gnn_k8.pt --set seed=1 output_dir=outputs/sac_seeds/<tên>_seed1` (khoảng 1 giờ). Nhớ truyền cùng các `--set` kênh như bản gốc (ví dụ `system.min_distance_m`).
- **Chấm điểm:** `python scripts/paper_scorecard.py --config configs/default.yaml configs/calibrated.yaml [--set output_dir=...]`. Lệnh này ghi đè `scorecard.json` trong thư mục output.
- **Tài liệu:** sửa `docs/history.json` rồi chạy `python scripts/export_docs.py`, sẽ sinh lại MODEL_HISTORY.md, PARAMETERS.md, parameters.json. Đừng sửa tay ba file này.
- **Công cụ khác:** `study_gnn_optimum.py`, `study_deployment_sensitivity.py`, `exp_beamforming.py --only users power noise --no-mlp --gnn <ckpt>` (chỉ đánh giá), `plot_layout.py`.

## 8. Lưu ý dễ vấp

- **Thư mục `outputs/`** (đã gitignore, khoảng 2 GB):

  | Thư mục | Nội dung |
  |---|---|
  | `runs_calibrated/` | R15 |
  | `runs/` | profile Table I nguyên văn, lỗi thời |
  | `archive_r*/` | R7, R9-R14, R13b, R14_s2, R15a, R15b |
  | `sac_seeds/` | r14/r15/r15b_seed1, seed2 |
  | `studies_r14/`, `studies_r15/` | sàng lọc, không có checkpoint ở local; checkpoint ở server |

  Tên R-số chỉ có trong docs, không có trong tên thư mục.
- **Ổ server:** `/home` đầy 96% (còn khoảng 150 GB). Có thể dọn `archive_v2`, `archive_h128`, `archive_v3`, `quick` nếu cần chỗ.
- **Mất kết nối SSH:** server có thể vẫn ping được trong khi cổng 22 timeout, do mạng hoặc VPN. Phiên tmux vẫn chạy tiếp. Vòng chờ phải coi "không kết nối được" khác với "đã xong".
- **Windows Git Bash:** heredoc dài có dấu nháy hay bị hiểu sai. Viết lệnh phức tạp vào file script rồi chạy (`ssh host "bash -s" < script.sh`). Không dùng `sleep` cố định để chờ.
- **Line ending:** repo dùng LF. Sửa file bằng Python `read_text` rồi `write_bytes`, hoặc dùng Edit.
- **`latest_run`** (`src/uavsec/utils.py`) chọn lần chạy mới nhất đã có `metrics.json`. Thư mục `deployment_sensitivity_*` cũng khớp mẫu `deployment_*`, nên khi tự glob hãy dùng `deployment_2*`.
- **Số liệu paper** là số đọc từ hình (`configs/paper_reference.yaml`, 400 dpi), chỉ để so sánh. Không gán số tuning của dự án cho paper.
- **Checkpoint cũ** nạp được với code mới: các khóa kiến trúc mới mặc định giữ hành vi cũ khi không có trong `hparams`.

## 9. Bản đồ file

| Cần gì | Ở đâu |
|---|---|
| Lịch sử, bảng xếp hạng, bảng sàng lọc, so với paper | `docs/MODEL_HISTORY.md` (nguồn: `docs/history.json`) |
| Tham số hiện tại, nguồn gốc, các giá trị đã thử | `docs/PARAMETERS.md`, `docs/parameters.json` |
| Thông số paper đã chuẩn hóa, số trang, bản đồ phương trình | `source_materials/paper-parameters.md` (gitignore, chỉ ở local) |
| Yêu cầu môn học | `source_materials/course-requirements.md` |
| Cấu hình | `configs/default.yaml`, `configs/calibrated.yaml`, `configs/paper_reference.yaml`, `configs/layouts/` |
| Mô hình kênh, rate, GNN | `src/uavsec/channel.py`, `src/uavsec/secrecy.py`, `src/uavsec/models/` |
| Train, baseline, môi trường SAC | `src/uavsec/training/`, `src/uavsec/baselines/`, `src/uavsec/envs/`, `src/uavsec/rl/` |
| Điểm vào chạy | `scripts/exp_*.py`, `scripts/make_report.py`, `scripts/make_animation.py`, `scripts/paper_scorecard.py`, `scripts/summarize_screening.py`, `scripts/server/` |
