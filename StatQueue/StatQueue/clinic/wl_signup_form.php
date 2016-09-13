<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if (isset($_GET['doctorid'])){
	display_wl_signup_form($_GET['clinicid'], $_GET['doctorid']);
} else if (isset($_GET['clinicid'])){
	display_wl_signup_form($_GET['clinicid']);
} 
do_footer();