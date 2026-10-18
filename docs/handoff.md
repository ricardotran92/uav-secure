# Handoff: tái hiện Tang et al. (TMC 2025), trạng thái đến 2026-10-19

Dành cho: người (hoặc phiên làm việc) tiếp quản dự án. Đọc xong file này là đủ để chạy lại, hiểu các quyết định và biết việc còn dở. Chi tiết từng thay đổi nằm ở [MODEL_HISTORY.md](MODEL_HISTORY.md), bảng tham số ở [PARAMETERS.md](PARAMETERS.md).

## 1. Tình trạng một đoạn

Code PyTorch tái hiện toàn bộ pipeline của paper (GNN beamforming, SAC deployment, so sánh với MLP và OPT, báo cáo tự động, animation) đã chạy end-to-end trên GPU. Kết quả chính là **R14**, điểm so với paper **0.097** (`scripts/paper_scorecard.py`, càng thấp càng khớp; R13 là 0.109; bản Table I nguyên văn khoảng 2.4). 16 thông số Table I và 5 lớp GCN giữ nguyên (kiểm tự động, 0 lệch). 35 test đều qua. **Chưa có gì được push lên GitHub**: hơn 50 commit chỉ ở máy local, `origin` chưa có nhánh nào.

## 2. Kết quả hiện tại (R14, `outputs/runs_calibrated/`)

R14 = R13 + LayerNorm và kết nối tắt trong mỗi lớp GCN + đầu vào GNN không nhân theo công suất test (C37, C40). So với R13, điểm giảm ở Fig. 4/5/6/9a/9b/9c/10, Fig. 7 giữ nguyên, Fig. 8 kém hơn một chút (0.207 → 0.225). Bảng đầy đủ ở đầu MODEL_HISTORY.md. Số đọc từ hình paper có sai số khoảng ±5%.

| Hạng mục | R14 | R13 | Paper | |
|---|---|---|---|---|
| OPT vs công suất, Fig. 9b (0 / 20 / 40 dBm) | 0.94 / 7.5 / 26.4 | như R14 | 1.0 / 7.5 / 28.0 | khớp |
| OPT K=8, Fig. 9a | 15.0 | 15.0 | 15.5 | khớp (OPT là bộ giải thay thế của mình) |
| Điểm tối ưu Fig. 7 so với (56, 140) | 6.3 m | 6.3 m | 0 | khớp |
| Graph-DRL, Fig. 8 | 23.8 bps/Hz | 23.3 | 23 | khớp |
| SAC dừng cách điểm tối ưu | 3.4 m (1 seed) | 3.3 / 5.4 / 8.6 m | khoảng 3 m | khớp |
| GNN K=8 / GNN so với OPT | **14.0 / 0.94** | 13.5 / 0.90 | 14.8 / 0.95 | khớp |
| GNN vs công suất (10 / 20 / 30 / 40 dBm) | 2.2 / 6.7 / 14.0 / 21.0 | 2.9 / 7.1 / 13.5 / 20.2 | 2.0 / 7.0 / 14.6 / 23.8 | gần (40 dBm: 0.80 so với OPT, paper 0.85) |
| GNN ở K=1, so với OPT | **0.35** (1.27 bps/Hz) | 0.69 | 0.91 (3.86) | **kém hơn R13 và paper** |
| Reward SAC cuối, Fig. 6 | 4110 / 23.5 | 4018 / 23.0 | 4400 / 25 | thấp hơn 7% |
| Loss GNN cuối, Fig. 4 | -13.7 | -13.2 | -14.9 | gần |
| MLP K=8 | 9.63 | 9.63 | 5.4 | **mạnh hơn paper** (không làm yếu đi) |
| Heuristic Fig. 8 so với Graph-DRL | 0.68-0.85 (+18%) | 0.67-0.81 (+24%) | 0.39-0.61 (+64%) | khoảng cách nhỏ hơn paper |

Ba điểm đáng nói với thầy: (1) phần "mạnh hơn" của MLP và heuristic làm lợi thế GNN trông nhỏ hơn paper, không phải thắng thật; (2) các chỉ số khớp dựa trên cách hiểu kênh đã hiệu chỉnh (mục 4), không phải giá trị paper công bố; (3) R14 đổi K=1 lấy K≥2: GNN có LayerNorm ngoại suy kém sang đồ thị chỉ có một nút (không có hàng xóm). Fig. 9a vẫn tốt hơn tổng thể vì 10 điểm K còn lại đều tăng. Bản R14-S2 (`outputs/archive_r14_s2/`, điểm 0.105) giữ K=1 và Fig. 8 như R13.

## 3. Cách chạy lại

- **Môi trường local:** Python 3.10 (cài `pip install -e ".[dev]"`), `python -m pytest`. Chạy thử nhanh mọi script bằng `--quick`.
- **Server:** `ricardo@140.123.104.113`, conda env `uavsec` (torch 2.5.1+cu121; **không nâng torch**, driver 535 không chạy được bản mới hơn), GPU 2 và 3 đều được dùng (bạn đã đăng ký thêm GPU 3). Hiện không có phiên tmux nào đang chạy.
- **Đẩy code / kéo dữ liệu:** `bash scripts/server/sync.sh push` (tar qua ssh, không có rsync). Kéo dữ liệu bằng `ssh ... tar -czf - outputs/<thư mục>` rồi `tar -xzf -` ở local.
- **Chạy đầy đủ một profile:** `GPU=2 bash scripts/server/run_all.sh --config configs/default.yaml configs/calibrated.yaml --set output_dir=outputs/runs_xxx` (khoảng 4-5 giờ). Profile Table I nguyên văn: bỏ `--config`. `scripts/server/run_deployment.sh` chỉ chạy Phase 4-5.
- **Chấm điểm:** `python scripts/paper_scorecard.py --config configs/default.yaml configs/calibrated.yaml`. Lưu ý lệnh này ghi đè `scorecard.json` trong thư mục output.
- **Tài liệu:** sửa `docs/history.json` rồi `python scripts/export_docs.py` để sinh lại MODEL_HISTORY.md, PARAMETERS.md, parameters.json (đừng sửa tay ba file này).
- **Công cụ nghiên cứu:** `study_gnn_optimum.py` (sàng lọc GNN, 1 seed khoảng 11 phút; `scripts/server/screen_gnn.sh` gói sẵn 3 seed và đánh giá Fig. 9a/9b), `study_deployment_sensitivity.py` (so sánh cách hiểu kênh), `exp_beamforming.py --only users power noise --no-mlp --gnn <ckpt>` (chỉ đánh giá), `plot_layout.py`.

## 4. Cấu hình hiện tại và cách đọc

Giá trị mặc định nằm ở `configs/default.yaml`; profile chính chồng thêm `configs/calibrated.yaml`. Mỗi giá trị không phải của paper đều có chú thích `# assumption`.

- **Của paper, giữ nguyên:** vùng 200×200 m, 8 user/eve, 8 anten, độ cao 100 m, Rician 10 dB, 1 Hz, 1 W, nhiễu 1.2e-13 W, GNN lr 0.005 / 300 episode / batch 512 / 5 lớp, SAC lr 3e-4 / 500 episode / mini-batch 64 / buffer 1e6 / discount 0.99, khoảng hành động [-1, 1].
- **Hiệu chỉnh (không phải giá trị paper):** path loss `30 + 22 ln(d)`, khoảng cách ngang, khoảng cách tối thiểu **30 m**. Paper chỉ ghi "log(d)" và không nói cơ số hay loại khoảng cách. Cách hiểu thông thường (log10, 3-D) cho bản đồ rate gần như phẳng, không tái tạo được Fig. 7/8.
- **Giả định hiện dùng:** GNN lớp ẩn 512, LayerNorm trước mỗi PReLU của f_gen/f_out và kết nối tắt e_k ← e_k + f_out(c_k) (`gnn.layer_norm=layer`, `gnn.residual=true`), đặc trưng nút cố định ở P^max (`gnn.input_power_scaling=false`; công suất test chỉ vào qua chuẩn hóa W), 150 mini-batch mỗi episode, LR cosine bắt đầu từ 0.005, Adam, cắt gradient 1.0, **train với UAV đúng tại tâm vùng** (đúng câu "this is the setup to train the GNN model" của paper); SAC 2 gradient step mỗi bước, bước bay 2.5 m, 175 bước mỗi episode, trạng thái nén log, không scale reward, chọn policy tốt nhất qua đánh giá 3 episode; tìm điểm tối ưu bằng GNN với 128 mẫu fading.
- **Layout Fig. 7:** số hóa từ hình (±1 m), `configs/layouts/paper_fig7.yaml`. Cách ghép eve với user là suy luận (paper không nêu). Cặp U5/U6 mơ hồ; bản đảo cặp `paper_fig7_swapped.yaml` chấm tốt hơn (0.145 so với 0.157 trên R7) nhưng người dùng chọn giữ cặp gốc.

## 5. Các quyết định đã chốt

| Quyết định | Lý do |
|---|---|
| Giữ nguyên mọi giá trị Table I; chỉ đổi chi tiết paper không công bố | yêu cầu của người dùng; kiểm tự động bằng `export_docs.py` |
| Không làm yếu MLP hoặc OPT để số liệu đẹp hơn | tránh gian lận số liệu; thầy hỏi một câu là lộ |
| Nhận một thay đổi chỉ khi điểm tổng scorecard giảm (và điểm tối ưu Fig. 7 trung vị ≤ 20 m); khi hòa thì chọn cách đọc sát nghĩa đen của paper | một tiêu chí khách quan, không chọn hình nào đẹp thì lấy hình đó |
| Chấp nhận lịch LR cosine bắt đầu từ 0.005 | người dùng duyệt ngày 2026-10-14; paper không nói lịch LR |
| Dùng cách hiểu kênh đã hiệu chỉnh | người dùng chọn sau khi xem nghiên cứu độ nhạy (8 cách hiểu) |
| Giữ cặp U5-E5, U6-E6 (không đảo) | người dùng chọn R7 làm chuẩn; đảo cặp chỉ ghi lại như một kết quả phụ |
| Không push GitHub khi chưa được hỏi | quy ước của phiên làm việc |
| Điểm tối ưu Fig. 7 tìm bằng GNN (đúng ràng buộc (24) của bài toán (23)-(25)); điểm tối ưu bằng OPT ghi kèm để tham khảo | bám đúng paper |

**Đã thử và loại** (số liệu trong MODEL_HISTORY.md, mục C-xx): paper SNR (nhiễu +30 dB), UAV rải đều khi train, UAV lệch quanh tâm, khởi tạo MRT, f_out 3 lớp, nén log đầu vào, lớp ẩn 768/1024 (sụp đổ ở lr 0.005), SGD thuần momentum, quy tắc K=1 "self" khi suy luận, bước bay 1 m (SAC không tới được điểm tối ưu), eve cách 20 ± 5-10 m, khoảng cách tối thiểu 1/10/20/50/100 m (30 m là mốc tốt nhất; 25 m ngang nhau, lưu làm R13b); vòng R14 (C36-C41, `outputs/studies_r14/`): aggregate gồm chính nút, chỉ LayerNorm, chỉ kết nối tắt, quy tắc K=1 "self" khi suy luận trên mạng R14.

## 6. Việc còn dở

**A. Cải thiện GNN.** Ba hướng 1-3 của handoff trước đã thử ở vòng R14 (2026-10-18/19): hướng 2 và 3 (LayerNorm cùng kết nối tắt) được nhận, hướng 1 bị loại. Việc còn lại, mỗi sàng lọc chạy `scripts/server/screen_gnn.sh <thư mục> <khóa>=<giá trị>` (3 seed và đánh giá Fig. 9a/9b, khoảng 70 phút một GPU):
1. Sửa K=1 của R14 (0.35 so với OPT, paper 0.91) mà không mất K≥2. Hướng thử: LayerNorm chỉ trong f_out (bỏ ở f_gen), hoặc chuẩn hóa không phụ thuộc giá trị aggregate (ví dụ chỉ chuẩn hóa nhánh m_k trước khi concat).
2. Tăng lên 400 mini-batch mỗi episode (lợi ích có thể nhỏ).
3. Cân bằng lại Fig. 6/8: lợi thế Graph-DRL so với heuristic tốt nhất còn +18% (paper +64%), vì GNN tốt hơn cũng nâng điểm geometric centre. Khoảng cách tối thiểu 25 m (R13b) từng cho +33%.
4. Chạy thêm 2 seed SAC cho R14 (hiện mới có 1 seed).

Nếu thấy cải thiện không đáng, có thể dừng: điểm 0.097 đã gần mức sai số số hóa, chạy tiếp có thể chỉ là chạy theo nhiễu.

**B. Sản phẩm nộp cho môn học (chưa làm).** Syllabus yêu cầu báo cáo mô phỏng (Word/PDF) và slide kết quả bằng tiếng Anh. Nguồn có sẵn: `outputs/runs_calibrated/report/report.md` (kèm hình PNG/PDF), animation `outputs/runs_calibrated/deployment_*/animation/uav_trajectory.gif`, bảng so sánh với paper trong MODEL_HISTORY.md. Chưa có file Word/PDF hay slide nào.

**C. Việc nhỏ chưa xong:**
- Push các commit local lên `origin` (cần người dùng đồng ý và có quyền).
- Fig. 11 (thời gian chạy) chưa số hóa và chưa chấm điểm: GNN nhanh hơn OPT khoảng 1500 lần (2.4 ms so với 3.7 s), nhưng thời gian OPT của mình không tăng theo K như paper; phụ thuộc phần cứng.
- Bonus cải tiến (ví dụ GNN chịu sai số CSI của eavesdropper) đã hoãn theo lựa chọn ban đầu của người dùng.
- Mới chạy 1 seed SAC cho mỗi cấu hình chính thức ngoài 3 seed xác nhận; số seed GNN là 3.
- Cặp user-eve của layout Fig. 7 vẫn là suy luận; nên ghi rõ trong báo cáo.

## 7. Lưu ý dễ vấp

- **Quy ước thư mục `outputs/`** (đã gitignore, khoảng 1.5 GB): `runs_calibrated/` = profile chính (R14), `runs/` = profile Table I nguyên văn (chạy ở vòng R12 với mặc định GNN cũ; **R14 đổi `default.yaml` nên `runs/` đã lỗi thời**, cần chạy lại nếu dùng; báo cáo của nó ở `outputs/report/`), `archive_r*/` = các bản cũ (R7, R9, R10, R11, R12, R13, R13b, R14-S2 ở `archive_r14_s2`), `tune_*` và `studies*` = thí nghiệm (`studies_r14/` = sàng lọc vòng R14, không có checkpoint ở local). Tên R-số chỉ có trong docs, không có trong tên thư mục.
- **Server còn các bản cũ chưa kéo về:** `archive_v2`, `archive_h128`, `archive_v3`, `quick` (khoảng 600 MB). Ổ `/home` của server đã đầy 96% (còn 154 GB); dọn các `archive_*` cũ nếu cần chỗ.
- **Windows Git Bash:** heredoc dài có dấu nháy hay bị hiểu sai; viết lệnh phức tạp vào file script rồi chạy. Không dùng `sleep` cố định để chờ, hãy dùng vòng `until` kiểm tra điều kiện.
- **Line ending:** repo dùng LF (`.gitattributes`); sửa file bằng Python `read_text` rồi `write_bytes` để không sinh CRLF.
- **`latest_run`** (trong `src/uavsec/utils.py`) chọn lần chạy mới nhất đã có `metrics.json`, khớp tên kèm timestamp chính xác; chạy lại nhiều lần trong cùng thư mục thì báo cáo và scorecard luôn lấy lần mới nhất.
- **Số liệu paper là số đọc từ hình** (`configs/paper_reference.yaml`, độ phân giải 400 dpi): chỉ để so sánh, không mô hình nào đọc file này. Không gán các số tuning hay ước lượng của dự án cho paper.
- **Lỗi đã gặp và đã sửa trong tài liệu:** snapshot R13 từng ghi MLP K=8 là 8.80 (sai); giá trị đúng 9.63, đã sửa trong `history.json`.

## 8. Bản đồ file

| Cần gì | Ở đâu |
|---|---|
| Lịch sử, bảng xếp hạng, so sánh với paper | `docs/MODEL_HISTORY.md` (nguồn: `docs/history.json`) |
| Tham số hiện tại, nguồn gốc từng tham số, các giá trị đã thử | `docs/PARAMETERS.md`, `docs/parameters.json` |
| Thông số của paper đã chuẩn hóa, kèm số trang và bản đồ phương trình | `source_materials/paper-parameters.md` (thư mục bị gitignore, chỉ có ở máy local) |
| Cấu hình | `configs/default.yaml`, `configs/calibrated.yaml`, `configs/paper_reference.yaml`, `configs/layouts/` |
| Mã nguồn | `src/uavsec/` (channel, secrecy, models, training, baselines, envs, rl, evaluation, scorecard, report) |
| Điểm vào chạy | `scripts/exp_*.py`, `scripts/make_report.py`, `scripts/make_animation.py`, `scripts/paper_scorecard.py`, `scripts/server/` |
