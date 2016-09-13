<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if (count($_POST) == 0){
	display_forgot_form();
} else {
	if (!filled_out($_POST)){
		throw new Exception("Both fields must be filled.");
	}
	$userid = $_POST['userid'];
	$email = $_POST['email'];
	$new_pw = reset_pw($userid);
	notify_pw($userid, $email, $new_pw);
	echo '<p style="text-align:center"><br/>Your new password has been emailed to you.<br /></p>';
}
do_footer();
?>
