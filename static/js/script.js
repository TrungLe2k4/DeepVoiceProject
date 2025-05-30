document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let formData = new FormData();
    formData.append("audio_file", document.getElementById("audioFile").files[0]);

    // Hiển thị "Loading..."
    document.getElementById("result").style.display = "none";
    document.getElementById("svmResultMfcc").textContent = "Loading...";
    document.getElementById("rfResultMfcc").textContent = "Loading...";
    document.getElementById("xgbResultMfcc").textContent = "Loading...";

    // Gửi yêu cầu POST lên server
    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    // Nhận kết quả và hiển thị
    const data = await response.json();
    document.getElementById("svmResultMfcc").textContent = data.svm_label_mfcc;
    document.getElementById("rfResultMfcc").textContent = data.rf_label_mfcc;
    document.getElementById("xgbResultMfcc").textContent = data.xgb_label_mfcc;

    document.getElementById("svmResultLogmel").textContent = data.svm_label_logmel;
    document.getElementById("rfResultLogmel").textContent = data.rf_label_logmel;
    document.getElementById("xgbResultLogmel").textContent = data.xgb_label_logmel;

    document.getElementById("svmResultVgg").textContent = data.svm_label_vgg;
    document.getElementById("rfResultVgg").textContent = data.rf_label_vgg;
    document.getElementById("xgbResultVgg").textContent = data.xgb_label_vgg;

    document.getElementById("result").style.display = "block";
});
