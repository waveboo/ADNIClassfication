$(document).ready(function(){
    var smripath = "";
    var fmripath = "";
    $("#fmri").hide();
    $("#classification-model").change(function () {
        let model = $("#classification-model").val();
        if(model == "2"){
            $("#fmri").show()
        }
        else{
            $("#fmri").hide()
        }
    })
    $("#dig").click(function () {
        let type = $("#classification-type").val();
        let model = $("#classification-model").val();
        if(model == "1"){
            let file_obj = $('#smri-data').get(0).files[0];
            // 将文件对象打包成form表单类型的数据
            let formdata = new FormData;
            formdata.append('file',file_obj);
            formdata.append('dtype', type);
            formdata.append('mode', model);
            var xhr = new XMLHttpRequest();
            xhr.open("post", "http://127.0.0.1:5000/upload_smri", true);
            xhr.onload = function () {
                console.log("连接完成!");
            };
            xhr.send(formdata);
            xhr.onreadystatechange = function(){
                //若响应完成且请求成功
                if(xhr.readyState === 4 && xhr.status === 200){
                    res = $.parseJSON(xhr.responseText);
                    if(res['code'] == '200'){
                        window.location.href = "report.html?filename=" + res['filename'] + "&mode=" + model + "&type=" + type
                    }
                }
            }
        }
        else if(model == "2"){
            let file_obj1 = $('#smri-data').get(0).files[0];
            // 将文件对象打包成form表单类型的数据
            let formdata1 = new FormData;
            formdata1.append('file',file_obj1);
            formdata1.append('dtype', type);
            formdata1.append('mode', model);

            var xhr1 = new XMLHttpRequest();
            xhr1.open("post", "http://127.0.0.1:5000/upload_smri", true);
            xhr1.onload = function () {
                console.log("连接完成!");
            };
            xhr1.send(formdata1);
            xhr1.onreadystatechange = function(){
                //若响应完成且请求成功
                if(xhr1.readyState === 4 && xhr1.status === 200){
                    let file_obj2 = $('#fmri-data').get(0).files[0];
                    let formdata2 = new FormData;
                    formdata2.append('file',file_obj2);

                    var xhr2 = new XMLHttpRequest();
                    xhr2.open("post", "http://127.0.0.1:5000/upload_fmri", true);
                    xhr2.onload = function () {
                        alert("连接完成!");
                    };
                    xhr2.send(formdata2);
                    xhr2.onreadystatechange = function(){
                        res = $.parseJSON(xhr2.responseText);
                        if(res['code'] == '200'){
                            window.location.href = "report.html?filename=" + $.parseJSON(xhr1.responseText)['filename'] + "&mode=" + model + "&type=" + type
                        }
                    }
                }
            }


        }
    });
});