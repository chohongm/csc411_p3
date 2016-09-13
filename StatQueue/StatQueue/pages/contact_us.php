<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if (isset($_REQUEST['user_email']))  {

	$master_email = "hongman.cho@gmail.com";
	$user_email = "From: ".$_REQUEST['user_email'];
	$title = $_REQUEST['title'];
	$message = $_REQUEST['message'];

	mail($master_email, "$title", $message, $user_email);

	echo "<p style='text-align:center'>Thank you for contacting us.</br>
		   We will send a reply to your email shortly.</p>";
} else {
	display_email_form();
}

do_footer();
?>