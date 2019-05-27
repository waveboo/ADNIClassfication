function GetRequest() {
    var url = location.search; //获取url中"?"符后的字串
    var theRequest = new Object();
    if (url.indexOf("?") != -1) {
        var str = url.substr(1);
        strs = str.split("&");
        for(var i = 0; i < strs.length; i ++) {
            theRequest[strs[i].split("=")[0]]=decodeURI(strs[i].split("=")[1]);
        }
    }
    return theRequest;
}

function toPercent(point){
    var str=Number(point*100).toFixed(2);
    str+="%";
    return str;
}

function doPrint() {
    var bodyContent=document.body.innerHTML;
    var printContent = document.getElementById("toprint").innerHTML;
    window.document.body.innerHTML=printContent;
    window.print();
    document.body.innerHTML=bodyContent;
}

$(document).ready(function(){
    $('introduction').hide();
    //调用
    var Request = new Object();
    Request = GetRequest();
    var filename = Request['filename'];
    var mode = Request['mode'];
    var type = Request['type'];

    var suburl = 'single_mode';
    if(mode == '2'){
        suburl = 'muti_mode';
    }

    $.ajax({
        type: "POST",
        url: "http://127.0.0.1:5000/" + suburl,
        data: {filename: filename, type: type},
        dataType: "json",
        success: function(res){
            let date = new Date()
            $('#time').html(date.getFullYear() + '年' + (parseInt(date.getMonth()) + 1) + '月' + date.getDate() + '日');
            $('#type').html(res['type']);
            $('#score').html(toPercent(res['score']));
            $('#patient').html(res['patient']);
            $('#id').html(res['id']);
            $('#date').html(res['date']);
            $('#tp').html(res['tp']);
            $('#visit').html(res['visit']);
            $('#sex').html(res['sex'] == 'M'?'男':'女');
            $('#age').html(res['age']);
            $('#dtype').html(res['dtype']);
            $('#dinfo').html(res['dinfo']);
            $('#format').html(res['format']);
            $("#print").click(function () {
                doPrint()
            });
            $('#preloader').delay(100).fadeOut('slow',function(){$(this).remove();});
            $('introduction').show();
        },
        error: function (err) {
            console.log(err)
        }
    });

});
