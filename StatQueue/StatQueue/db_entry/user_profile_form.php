<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$db = db_connect();

$my_profile = $_SESSION['my_profile'];
$my_profile->db = $db;
$my_profile->display_profile_edit_form();

do_footer();
?>