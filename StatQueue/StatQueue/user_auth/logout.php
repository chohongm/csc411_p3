<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header(false);
  
  $user = check_valid_user();
  
  if ($user == 'admin'){
  	$clinic = $_SESSION['saved_clinic'];
  	$clinic->db = db_connect();
  	$clinic->reset_avail_doctors();
  }
  
  foreach($_SESSION as $key => $value){
  	unset($_SESSION[$key]);
  }
  session_destroy();
  echo "<br /><br /><h3 style='text-align: center'>You are now logged out.</h3>";
  #echo "<p style = 'text-align: center'><span class='spec'> Now loading home page.. <br/></span></p>";
  #echo '<meta http-equiv="refresh" content= "2;URL= \'/pages/home.php\'">';
  do_footer();
?>

