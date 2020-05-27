var urlVal = "/visualize"
$(document).ready(function() {
    $('form').on('submit', function(event) {
        if(document.getElementById("radio-b").checked == true)
            urlVal= "/visualize"
        else if(document.getElementById("radio-a").checked == true)
            urlVal = "/viewRaw"
      $.ajax({
            data : $('form').serialize(),
            type : 'POST',
            url : urlVal
           })
       .done(function(data) {
         $('#vizoutput').html(data);
     });
     event.preventDefault();
     });
});
$(function () {
    $('#radio-b').change(function(){
        $('form :input').val('');
        $('#visualize').val('Submit') 
    });
    $('#radio-a').change(function(){
        $('form :input').val('');
        $('#visualize').val('Submit')
    });
});