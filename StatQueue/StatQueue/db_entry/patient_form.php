<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();

  if (check_valid_user() == 'admin' || check_valid_user() == 'master'){
  	display_patient_form();
  } else if (check_valid_user() == 'user'){
	echo "<br /><br /><p>Sorry, update waitlist option is only accessible to the registered clinic administrators.</p>";
  } else {
	echo "<br /><br /><p>Sorry, update waitlist option is only accessible to the registered clinic administrators.
		  <br /><br />If you are an administrator and already have your Touch In account,
		  please log in to access this page.</p>";
	echo "<a href='/user_auth/login.php'>click to log in</a>";
  }
  do_footer();
?>