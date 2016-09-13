<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$db = db_connect();

$patient_profile = $_SESSION['patient_profile'];
$patient_profile->db = $db;
$patient_profile->display_profile_edit_form();

do_footer();
?>