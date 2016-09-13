<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");

if (@store_new_post($_POST) == 'true'){
	#if storing succeeds..
	include ('index.php');
} else {
	$error = true;
	include ('post_form.php');
}
?>