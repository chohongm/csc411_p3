<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();

  if (check_valid_user()){
  	$db = db_connect();
  	$userid = $_SESSION[check_valid_user()];
  	$postalcode = user::get_postalcode($db, $userid);
  	display_search_form($postalcode);
  } else {
  	echo "<br /><br /><p>This page is only available to our users.
     	  Please log in to your account first.<p>";
  	echo "<a href='/user_auth/login.php?url=search_form'>click to log in</a>";
  }
  do_footer();
?>