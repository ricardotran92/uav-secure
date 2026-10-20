# Handoff: tái hiện Tang et al. (TMC 2025), trạng thái đến 2026-10-20

Dành cho: người (hoặc phiên làm việc) tiếp quản dự án. Đọc xong file này là đủ để chạy lại, hiểu các quyết định và biết việc còn dở. Chi tiết từng thay đổi nằm ở [MODEL_HISTORY.md](MODEL_HISTORY.md), bảng tham số ở [PARAMETERS.md](PARAMETERS.md).

## 1. Tình trạng một đoạn

Code PyTorch tái hiện toàn bộ pipeline của paper (GNN beamforming, SAC deployment, so sánh với MLP và OPT, báo cáo tự động, animation) đã chạy end-to-end trên GPU. Kết quả chính là **R15**, điểm so với paper **0.083** với 1 seed SAC, **0.095** khi lấy trung bình 3 seed SAC (`scripts/paper_scorecard.py`, càng thấp càng khớp; R14 0.097 / 0.0985, R13 0.109; bản Table I nguyên văn khoảng 2.4). Từ R15 các bản ứng viên được so bằng điểm trung bình 3 seed SAC (C47). 16 thông số Table I và 5 lớp GCN giữ nguyên (kiểm tự động, 0 lệch). 36 test đều qua. **Chưa có gì được push lên GitHub**: hơn 50 commit chỉ ở máy local, `origin` chưa có nhánh nào.

## 2. Kết quả hiện tại (R15, `outputs/runs_calibrated/`)

R15 = R14 + aggregate lấy max trên N(k) ∪ {k} + lớp ẩn 768 (C43, C44). Hai thay đổi này chỉ chạy tốt nhờ LayerNorm của R14: không có LayerNorm thì cả hai đều đã từng bị loại (C36, C33). Bảng xếp hạng, bảng sàng lọc (🥇🥈🥉) và bảng so với paper ở đầu MODEL_HISTORY.md. Số đọc từ hình paper có sai số khoảng ±5%.

| Hạng mục | R15 | R14 | Paper | |
|---|---|---|---|---|
| OPT vs công suất, Fig. 9b (0 / 20 / 40 dBm) | 0.94 / 7.5 / 26.4 | như R15 | 1.0 / 7.5 / 28.0 | khớp |
| Điểm tối ưu Fig. 7 so với (56, 140) | 6.3 m (3 seed GNN: 7.2 / 6.3 / 6.3) | 6.3 m | 0 | khớp |
| Graph-DRL, Fig. 8 | 23.7 bps/Hz | 23.8 | 23 | khớp |
| SAC dừng cách điểm tối ưu (3 seed) | 3.4 / 9.1 / 2.1 m | 3.4 / 8.3 / 2.2 m | khoảng 3 m | gần |
| GNN K=8 / so với OPT | 14.0 / 0.94 | 14.0 / 0.94 | 14.8 / 0.95 | khớp |
| GNN K=11 | 16.2 | 16.2 | 16.1 | khớp |
| GNN ở K=1 (so với OPT) | **2.93 (0.80)** | 1.27 (0.35) | 3.86 (0.91) | còn thấp hơn paper |
| GNN vs công suất (10 / 20 / 30 / 40 dBm) | 2.2 / 6.7 / 14.0 / 21.1 | 2.2 / 6.7 / 14.0 / 21.0 | 2.0 / 7.0 / 14.6 / 23.8 | gần (40 dBm 0.80, paper 0.85) |
| Reward SAC cuối, Fig. 6 (3 seed) | 4010-4108 / 23.5 | 4070-4123 | 4400 / 25 | thấp hơn 7-9% |
| Loss GNN cuối, Fig. 4 | -13.6 | -13.7 | -14.9 | gần |
| MLP K=8 | 9.63 | 9.63 | 5.4 | **mạnh hơn paper** (không làm yếu đi) |
| Heuristic Fig. 8 so với Graph-DRL | 0.69-0.87 (+15%) | 0.68-0.85 (+18%) | 0.39-0.61 (+64%) | khoảng cách nhỏ hơn paper |

So với R14: Fig. 9a 0.084 → 0.065, K=1 tăng gấp đôi; Fig. 4/5/8 kém một chút (0.079 → 0.089, 0.072 → 0.080, 0.225 → 0.236).

**Bản thay thế R15b** (`outputs/archive_r15b/`, R15 + khoảng cách tối thiểu 25 m): tốt nhất ở Fig. 4/5/6/8/9a/10 (reward 4205-4298, lợi thế Graph-DRL +23-26%, GNN K=8 14.3), nhưng Fig. 9b kém hẳn (OPT ở 0 dBm 1.35 so với paper 1.0). Trung bình 3 seed: 0.096 so với 0.095 của R15, tức gần như hòa. Nếu thầy coi trọng Fig. 6/8 hơn Fig. 9b thì R15b đáng trình bày.

Ba điểm đáng nói với thầy: (1) phần "mạnh hơn" của MLP và heuristic làm lợi thế GNN trông nhỏ hơn paper, không phải thắng thật; (2) các chỉ số khớp dựa trên cách hiểu kênh đã hiệu chỉnh (mục 4), không phải giá trị paper công bố; (3) điểm 1 seed của R15 (0.083) có phần may ở Fig. 7 SAC; con số nên trích là 0.095 (3 seed).

## 3. Cách chạy lại

- **Môi trường local:** Python 3.10 (cài `pip install -e ".[dev]"`), `python -m pytest`. Chạy thử nhanh mọi script bằng `--quick`.
- **Server:** `ricardo@140.123.104.113`, conda env `uavsec` (torch 2.5.1+cu121; **không nâng torch**, driver 535 không chạy được bản mới hơn), GPU 2 và 3 đều được dùng (bạn đã đăng ký thêm GPU 3). Hiện không có phiên tmux nào đang chạy.
- **Đẩy code / kéo dữ liệu:** `bash scripts/server/sync.sh push` (tar qua ssh, không có rsync). Kéo dữ liệu bằng `ssh ... tar -czf - outputs/<thư mục>` rồi `tar -xzf -` ở local.
- **Chạy đầy đủ một profile:** `GPU=2 bash scripts/server/run_all.sh --config configs/default.yaml configs/calibrated.yaml --set output_dir=outputs/runs_xxx` (khoảng 4-5 giờ). Profile Table I nguyên văn: bỏ `--config`. `scripts/server/run_deployment.sh` chỉ chạy Phase 4-5.
- **Chấm điểm:** `python scripts/paper_scorecard.py --config configs/default.yaml configs/calibrated.yaml`. Lưu ý lệnh này ghi đè `scorecard.json` trong thư mục output.
- **Tài liệu:** sửa `docs/history.json` rồi `python scripts/export_docs.py` để sinh lại MODEL_HISTORY.md, PARAMETERS.md, parameters.json (đừng sửa tay ba file này).
- **Công cụ nghiên cứu:** `study_gnn_optimum.py` (sàng lọc GNN, 1 seed khoảng 11 phút; `scripts/server/screen_gnn.sh` gói sẵn 3 seed và đánh giá Fig. 9a/9b, khoảng 1.5-4 giờ một biến thể; `scripts/summarize_screening.py <thư mục> --json` tổng hợp một vòng), thêm seed SAC: `exp_deployment.py --gnn <runs>/convergence_*/ckpt/gnn_k8.pt --set seed=1 output_dir=outputs/sac_seeds/<tên>` (khoảng 1 giờ), `study_deployment_sensitivity.py` (so sánh cách hiểu kênh), `exp_beamforming.py --only users power noise --no-mlp --gnn <ckpt>` (chỉ đánh giá), `plot_layout.py`.

## 4. Cấu hình hiện tại và cách đọc

Giá trị mặc định nằm ở `configs/default.yaml`; profile chính chồng thêm `configs/calibrated.yaml`. Mỗi giá trị không phải của paper đều có chú thích `# assumption`.

- **Của paper, giữ nguyên:** vùng 200×200 m, 8 user/eve, 8 anten, độ cao 100 m, Rician 10 dB, 1 Hz, 1 W, nhiễu 1.2e-13 W, GNN lr 0.005 / 300 episode / batch 512 / 5 lớp, SAC lr 3e-4 / 500 episode / mini-batch 64 / buffer 1e6 / discount 0.99, khoảng hành động [-1, 1].
- **Hiệu chỉnh (không phải giá trị paper):** path loss `30 + 22 ln(d)`, khoảng cách ngang, khoảng cách tối thiểu **30 m**. Paper chỉ ghi "log(d)" và không nói cơ số hay loại khoảng cách. Cách hiểu thông thường (log10, 3-D) cho bản đồ rate gần như phẳng, không tái tạo được Fig. 7/8.
- **Giả định hiện dùng:** GNN lớp ẩn 768, aggregate max trên N(k) ∪ {k} (`gnn.aggregate_self=true`), LayerNorm trước mỗi PReLU của f_gen/f_out và kết nối tắt e_k ← e_k + f_out(c_k) (`gnn.layer_norm=layer`, `gnn.residual=true`), đặc trưng nút cố định ở P^max (`gnn.input_power_scaling=false`; công suất test chỉ vào qua chuẩn hóa W), 150 mini-batch mỗi episode, LR cosine bắt đầu từ 0.005, Adam, cắt gradient 1.0, **train với UAV đúng tại tâm vùng** (đúng câu "this is the setup to train the GNN model" của paper); SAC 2 gradient step mỗi bước, bước bay 2.5 m, 175 bước mỗi episode, trạng thái nén log, không scale reward, chọn policy tốt nhất qua đánh giá 3 episode; tìm điểm tối ưu bằng GNN với 128 mẫu fading.
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

**Đã thử và loại** (số liệu trong MODEL_HISTORY.md, mục C-xx): paper SNR (nhiễu +30 dB), UAV rải đều khi train, UAV lệch quanh tâm, khởi tạo MRT, f_out 3 lớp, nén log đầu vào, lớp ẩn 768/1024 (sụp đổ ở lr 0.005), SGD thuần momentum, quy tắc K=1 "self" khi suy luận, bước bay 1 m (SAC không tới được điểm tối ưu), eve cách 20 ± 5-10 m, khoảng cách tối thiểu 1/10/20/50/100 m (30 m là mốc tốt nhất; 25 m ngang nhau, lưu làm R13b); vòng R14 (C36-C41, `outputs/studies_r14/`): aggregate gồm chính nút, chỉ LayerNorm, chỉ kết nối tắt, quy tắc K=1 "self" khi suy luận trên mạng R14; vòng R15 (C42-C46, `outputs/studies_r15/`): LayerNorm chỉ ở f_out, 300 mini-batch mỗi episode (lợi như lớp ẩn 768 nhưng chậm hơn), khoảng cách tối thiểu 25 m trên mạng R15 (lưu làm R15b).

## 6. Việc còn dở

**A. Cải thiện tiếp (nếu muốn).** Vòng R14 và R15 (2026-10-18 đến 10-20) đã làm hết các hướng của handoff cũ. Còn lại:
1. K=1 vẫn 0.80 so với OPT (paper 0.91) và 40 dBm 0.80 (paper 0.85).
2. Fig. 8/6: lợi thế Graph-DRL +15% (paper +64%). Chỉ khoảng cách tối thiểu 25 m kéo lên được (+26%, R15b), nhưng đổi lại Fig. 9b. Chưa tìm được một thay đổi thỏa cả hai phía.
3. Fig. 4: loss -13.6 so với -14.9. R15b đạt -14.0.

Nếu thấy cải thiện không đáng, có thể dừng. Chênh lệch giữa R15, R15b và R14 (0.095 / 0.096 / 0.0985 trên 3 seed) đã nhỏ ngang dao động giữa các seed SAC, nên chạy tiếp có thể chỉ là chạy theo nhiễu.

**B. Sản phẩm nộp cho môn học (chưa làm).** Syllabus yêu cầu báo cáo mô phỏng (Word/PDF) và slide kết quả bằng tiếng Anh. Nguồn có sẵn: `outputs/runs_calibrated/report/report.md` (kèm hình PNG/PDF), animation `outputs/runs_calibrated/deployment_*/animation/uav_trajectory.gif`, bảng so sánh với paper trong MODEL_HISTORY.md. Chưa có file Word/PDF hay slide nào.

**C. Việc nhỏ chưa xong:**
- Push các commit local lên `origin` (cần người dùng đồng ý và có quyền).
- Fig. 11 (thời gian chạy) chưa số hóa và chưa chấm điểm: GNN nhanh hơn OPT khoảng 1500 lần (2.4 ms so với 3.7 s), nhưng thời gian OPT của mình không tăng theo K như paper; phụ thuộc phần cứng.
- Bonus cải tiến (ví dụ GNN chịu sai số CSI của eavesdropper) đã hoãn theo lựa chọn ban đầu của người dùng.
- R14, R15, R15b đã có 3 seed SAC (`outputs/sac_seeds/`); R15a chỉ 1 seed. Số seed GNN là 3.
- Cặp user-eve của layout Fig. 7 vẫn là suy luận; nên ghi rõ trong báo cáo.

## 7. Lưu ý dễ vấp

- **Quy ước thư mục `outputs/`** (đã gitignore, khoảng 1.5 GB): `runs_calibrated/` = profile chính (R15), `runs/` = profile Table I nguyên văn (chạy ở vòng R12 với mặc định GNN cũ; **từ R14 `default.yaml` đã đổi nên `runs/` lỗi thời**, cần chạy lại nếu dùng; báo cáo của nó ở `outputs/report/`), `archive_r*/` = các bản cũ (R7, R9-R14, R13b, R14-S2, R15a, R15b), `sac_seeds/` = seed SAC thêm (r14/r15/r15b_seed1|2), `tune_*` và `studies*` = thí nghiệm (`studies_r14/`, `studies_r15/` = sàng lọc, không có checkpoint ở local). Tên R-số chỉ có trong docs, không có trong tên thư mục.
- **Server còn các bản cũ chưa kéo về:** `archive_v2`, `archive_h128`, `archive_v3`, `quick` (khoảng 600 MB). Ổ `/home` của server đã đầy 96% (còn khoảng 150 GB); dọn các `archive_*` cũ nếu cần chỗ.
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
