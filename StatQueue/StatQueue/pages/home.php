<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();
  #page content when already logged in 
  if (check_valid_user()){
    echo "<h3 style='text-align: center'>
    	  You are logged in!</h3><br />";
  } else {
?>
    <!-- page content when not logged in --><br />
	<h3 style="text-align: center">
	<a href="/user_auth/login.php"><img src="/images/login.png" border="0" width="75" height="30"></a>
	<a href="/db_entry/register_form.php"><img src="/images/signup.png" border="0" width="90" height="30"></a>
	</h3>
<?php
  }
  do_footer();
?>