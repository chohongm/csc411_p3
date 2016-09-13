<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");

if (@edit_post($_POST) == 'true'){
	#if editing succeeds..
	include ('index.php');
} else {
	$error = true;
	include ('post_form.php');
}
?>