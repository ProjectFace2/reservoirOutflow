
$(document).ready(function() {
    $('form').on('submit', function(event) {
      $.ajax({
            data : $('form').serialize(),
            type : 'POST',
            url : '/predict'
           })
       .done(function(data) {
         $('#output').html(data);
     });
     event.preventDefault();
     });
//      $.ajax({
//         type : 'POST',
//         url : '/plot.png'
//        })
//    .done(function(data) {
//      $('#images').html(data);
//  });
});
$(function () {
    $('#radio-b').change(function(){
        $('form :input').val('');
        $('#seconddate').css("display", "block");
        $('#click').val('Submit') 
    });
    $('#radio-a').change(function(){
        $('form :input').val('');
        $('#seconddate').css("display", "none");
        $('#click').val('Submit')
    });
});

