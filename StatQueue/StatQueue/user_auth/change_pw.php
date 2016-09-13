<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();

  @ $old_pw = $_POST['old_pw'];
  @ $new_pw = $_POST['new_pw'];
  @ $new_pw2 = $_POST['new_pw2'];

  if (!isset($old_pw) || !isset($new_pw) || !isset($new_pw2)){
  	display_pw_form();
  } else {
  	
  	#$user = check_valid_user();
  	$userid = $_SESSION[check_valid_user()];
  	
  	// attempt update
  	if(valid_pw($new_pw)){
  		if ($new_pw != $new_pw2) {
  			throw new Exception('Passwords entered were not the same.  Not changed.');
  		}
  		change_pw($userid, $old_pw, $new_pw);
  		echo 'Password changed.';
  	} else {
  		throw new Exception('New password must consist of at least one in each lowercase letter,
    						 uppercase letter, and number');
  	}
  }
  do_footer();
?>
